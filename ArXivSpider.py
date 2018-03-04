
# wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
# bash Anaconda2-4.3.1-Linux-x86_64.sh
# conda install -c anaconda feedparser=5.2.1
# conda install -c conda-forge mechanize=0.3.3


# nohup /home/amorandi/anaconda2/bin/python ArXivSpider.py </dev/null >/dev/null 2>&1 &  # for Hubble and cspar
# for i in uv dmc; do sed -e "s/cluster_pref=.*/cluster_pref=$i/g" asc_queue >/home/uahaxm/.asc_queue; sleep 3; run_script script.pbs; done  #for dmc and uv
# for i in /cspar /dmc /hubble; do yes | cp -a /mnt/hgfs/Shared\ Folders/Desktop/ArXivSpider.py $i/PyAndrea/.;done
# squeue | awk '/script.pbs/{print $1}' | xargs scancel; rm -f script.pbs.o*


import feedparser, urllib2, re, time, numpy as np, pickle, mechanize, platform,sys, datetime, os, pandas as pd, glob
from pandas.tseries.offsets import MonthBegin,MonthEnd
from bs4 import BeautifulSoup

#<editor-fold desc="grab">   to fold chuncks of codes in the IDE PyCharm
def grab(url):
    # Here we use mechanize to get the "Sort by citation drop-down menu"
    br = mechanize.Browser()
    br.set_handle_robots(False)
    resp = br.open(url)
    br.select_form(nr=0)
    control = br.form.find_control("qsort")
    for item in control.items:
        if "QSORT_CITES" in item.name:
            item.selected = True  # it matches "QSORT_CITES", so select it
            response = br.submit()
            break
    html=response.get_data()
    ref = re.findall(r"link_type=ABSTRACT\">(.*)</a>.*>([0-9]+.000).*>([0-9]+/[0-9]+)<", html)
    ref = [(BeautifulSoup(i[0], "lxml").get_text(), int(float(i[1])), i[2]) for i in ref if
            len(ref) > 0]  # for each publication: ADS Bibliographic Code, total citations, publication date
    return ref  #,resp.code
#</editor-fold>


# http://adsabs.harvard.edu/mirrors.html  list of ADS mirrors
mirrors_ads=['http://adsabs.harvard.edu/','http://cdsads.u-strasbg.fr/','http://esoads.eso.org/','http://ukads.nottingham.ac.uk/','http://ads.ari.uni-heidelberg.de/','http://ads.inasan.ru/','http://ads.astro.puc.cl/','http://ads.nao.ac.jp/','http://ads.bao.ac.cn/','http://ads.idia.ac.za/'];c0=0
mirrors_arxiv=['','lanl.','es.','in.','de.','cn.'];c=0
# if datetime.datetime.now().toordinal()==datetime.datetime(2017, 6, 10).toordinal(): mirrors_arxiv.pop(1) #lanl is down on this date
linux_platform=['dmc','uv']  #list of Linux servers

# see pag. 300 of "Python for Data Analysis" for Base Time Series Frequencies
window=24  #window of time (in months) to fetch data for each Linux server
date_start=pd.date_range('1/1/2013',periods=len(linux_platform),freq=str(window)+'MS').to_pydatetime()
date_final=(pd.date_range(date_start[0],periods=len(linux_platform),freq=str(window)+'MS')+window*MonthEnd()).to_pydatetime()
linux_id,linux_platform=[(k,i) for k,i in enumerate(linux_platform) if i in platform.node()][0] if platform.system() == 'Linux' else ('','')
date_start,date_final= (date_start[linux_id],date_final[linux_id]) if linux_platform else \
    (datetime.datetime(2010, 1, 1), datetime.datetime(2010, 12, 31))  # initial and final date of searching
if linux_platform=='hubble': mirrors_arxiv.pop(0)

t_sleep=datetime.timedelta(0,62)  #force the crawling to sleep for 0 minutes and xxx seconds
t_mirrors_ads=np.repeat(datetime.datetime.now(),len(mirrors_ads)); t_mirrors_arxiv=np.repeat(datetime.datetime.now(),len(mirrors_arxiv))

w=1 #window of time (in month(s)) to fetch data for each ArXiv query
data_dir='data'
if not os.path.exists(data_dir): os.makedirs(data_dir)
for i in glob.glob(data_dir+'/ArXiv_*.pkl'):
    with open(i, 'rb') as f:
        feed = pickle.load(f)
        if linux_platform in feed['Info']['platform node']:
            date_start = max(date_start, (feed['Info']['date_start']+w*MonthBegin()).to_pydatetime())

file_save='ArXiv_'+date_start.strftime('%Y%m%d') + '-' + date_final.strftime('%Y%m%d')+'.pkl'
while date_start <= date_final:
    date_end=(date_start+w*MonthEnd()).to_pydatetime()
    query='http://export.arxiv.org/api/query?search_query=cat:astro-ph*+AND+submittedDate:['+date_start.strftime('%Y%m%d')+'0000+TO+'+date_end.strftime('%Y%m%d')+'1159]&sortBy=submittedDate&sortOrder=ascending&max_results=10000'
    html = urllib2.urlopen(query).read()
    feed = feedparser.parse(html)
    feed.update({'Info':{'date_start':date_start, 'date_end':date_end, 'date_final':date_final, 'query':query,'t_sleep':t_sleep, 'platform node':platform.node(),'cpu_time_start':datetime.datetime.now()}})
    #
    for entry in feed.entries:
        ref=ref2=err_msg=publication_date=affiliations=biblio=[]; url_ArXiv=url_ads_cit=url_ads_ref=url_ads='';cit=-1
        c=(c+1) % len(mirrors_arxiv)
        try:  #crawling ArXiv
            url_ArXiv=entry['id'].replace('arxiv.org', mirrors_arxiv[c] + 'arxiv.org') #switch mirror
            while(datetime.datetime.now()-t_mirrors_arxiv[c])<t_sleep: time.sleep(0.4)
            t_mirrors_arxiv[c]=datetime.datetime.now()
            html = urllib2.urlopen(url_ArXiv).read() #fetch arXiv abstract page
            soup=BeautifulSoup(html,"lxml")
            url_ads=soup.find(attrs={'class': "extra-ref-cite"}).findAll('a')[-1].get('href') #ADS url
            #
            try: #fetching ADS abstract page
                c0 = (c0 + 1) % len(mirrors_ads)
                url_ads=url_ads.replace(mirrors_ads[0],mirrors_ads[c0])
                while (datetime.datetime.now() - t_mirrors_ads[c0]) < t_sleep: time.sleep(0.4)
                t_mirrors_ads[c0] = datetime.datetime.now()
                html = urllib2.urlopen(url_ads).read() #fetch ADS abstract page
                cit=re.findall(r"Citations to the Article \((\d+)\)", html) #citations
                cit = int(cit[0]) if cit else 0
                soup = BeautifulSoup(html, "lxml")
                body = soup.findAll(attrs={'align': 'left', 'valign': 'top'})
                for i, paragraph in enumerate(body):
                    if 'Affiliation:' in paragraph.text: affiliations=body[i+1].text.encode('ascii','ignore')
                    elif 'Publication Date:' in paragraph.text: publication_date=body[i+1].text
                    elif 'Bibliographic Code:' in paragraph.text: biblio=body[i+1].text
                if biblio: biblio = BeautifulSoup(biblio, "lxml").get_text()
            except Exception,e: # ssl.SSLError
                err_msg.append(['Error on line {}'.format(sys.exc_info()[-1].tb_lineno), str(e)])  #https://stackoverflow.com/questions/1278705/python-when-i-catch-an-exception-how-do-i-get-the-type-file-and-line-number
                raise Exception(e)
            #
            if cit>0: #fecthing ADS citations page (with page sorted by citations)
                try:
                    c0 = (c0 + 1) % len(mirrors_ads)
                    url_ads_cit=('http://adsabs.harvard.edu/cgi-bin/nph-ref_query?bibcode='+urllib2.quote(biblio)+'&amp;refs=CITATIONS&amp;db_key=AST').replace(mirrors_ads[0],mirrors_ads[c0])
                    while (datetime.datetime.now() - t_mirrors_ads[c0]) < t_sleep: time.sleep(0.4)
                    t_mirrors_ads[c0] = datetime.datetime.now()
                    ref = grab(url_ads_cit)  #for each publication citing the paper under consideration: ADS Bibliographic Code, total citations, publication date
                except Exception,e: # ssl.SSLError
                    err_msg.append(['Error on line {}'.format(sys.exc_info()[-1].tb_lineno), str(e)])
                    ref=[]
                    # raise Exception(e)
            #
            try:  # fecthing ADS page of references (with page sorted by citations)
                c0 = (c0 + 1) % len(mirrors_ads)
                url_ads_ref = ('http://adsabs.harvard.edu/cgi-bin/nph-ref_query?bibcode=' + urllib2.quote(biblio) + '&amp;refs=REFERENCES&amp;db_key=AST').replace(mirrors_ads[0], mirrors_ads[c0])
                while (datetime.datetime.now() - t_mirrors_ads[c0]) < t_sleep: time.sleep(0.4)
                t_mirrors_ads[c0] = datetime.datetime.now()
                ref2=grab(url_ads_ref)
            except Exception, e:  # ssl.SSLError
                err_msg.append(['Error on line {}'.format(sys.exc_info()[-1].tb_lineno),str(e)])
                ref2 = []
                # raise IOError(e)
        #
        except Exception, e:  # ssl.SSLError
            err_msg.append(['Error on line {}'.format(sys.exc_info()[-1].tb_lineno), str(e)])
            cit=-1; ref=ref2=[]; publication_date=affiliations=biblio=url_ads_ref=url_ads=url_ads_cit=''
        #
        entry.update({'ADS':{'Citations': cit, 'Bibliographic Code':biblio,'Affiliations':affiliations,'Publication Date':publication_date,'Citations list':ref,'References':ref2, 'url_ArXiv':url_ArXiv, 'url_ads':url_ads, 'url_ads_cit':url_ads_cit, 'url_ads_ref':url_ads_ref,'err_msg':err_msg}})  #updating the ArXiv feed with total citations, Bibliographic Code, citation and reference list
    #
    feed['Info']['cpu_time_end'] = datetime.datetime.now()
    with open(data_dir+'/ArXiv_' + date_start.strftime('%Y%m%d') + '-' + date_end.strftime('%Y%m%d') + '.pkl', 'wb') as f:
        pickle.dump(feed, f, protocol=-1) #one-time saving of 'feeds' as pickle
    # with open(file_save, 'ab') as f: #incremental saving of 'feeds'
    #     pickle.dump(feed, f, protocol=-1)
    #
    with open("log_"+linux_platform+".txt", "w") as text_file:
        text_file.write("%f \n" % (datetime.datetime.utcnow() - datetime.datetime.utcfromtimestamp(0)).total_seconds())
    date_start = (date_start+w*MonthBegin()).to_pydatetime()  #next w month(s) for getting arXiv papers

if platform.system() == 'Linux':
    if linux_platform != 'hubble':
        res = os.popen('echo AXivSpider.py finished execution in \'' + linux_platform + '\' | mail -s \'ArXivSpider\' andrea.m1020@gmail.com')
    sys.exit(0)  #exit on Linux

