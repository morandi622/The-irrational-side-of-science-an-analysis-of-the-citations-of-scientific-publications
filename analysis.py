
import matplotlib.pyplot as plt, pytz,os, glob
from dateutil.parser import parse
import numpy as np, pickle, datetime
import pymc3 as pm
import pandas as pd
import theano


# plt.ion()


os.chdir(r"\\vmware-host\Shared Folders\Shared Folders\Desktop\ArXivSpider_data")

# loading the data
obj=[]
for i in glob.glob('ArXiv_*.pkl'):
    with open(i, 'rb') as f:
        obj.append(pickle.load(f))



def time_diff(t):
#This function defines the time difference between submission time and cutoff time

    if t>datetime.datetime(2017,1,1): h=14
    else: h=16

    if (t.isoweekday() == 1 and t.hour < h):  # Monday before 16
        t_cutoff=datetime.datetime(t.year, t.month, t.day, h)-datetime.timedelta(3)
    elif 1 <= t.isoweekday() <= 5 and t.hour >= h:  # Monday to Friday after 16.00
        t_cutoff = datetime.datetime(t.year, t.month, t.day, h)
    elif 1 < t.isoweekday() <= 5 and t.hour < h:  # Tuesday to Friday, before 16.00
        t_cutoff = datetime.datetime(t.year, t.month, t.day, h) - datetime.timedelta(1)
    elif 5 < t.isoweekday():  # Saturday and Sunday
        t_cutoff=datetime.datetime(t.year, t.month, t.day, h)-datetime.timedelta(t.isoweekday()-5)

    return (t-t_cutoff).total_seconds(),t_cutoff.toordinal() # time difference between submission time and cutoff time and proleptic Gregorian ordinal of the date


#fetching citation and submission day
dt=[];cit=[]; to=[]
for feed in obj:
    for entry in feed.entries:
        t=parse(entry['published']).astimezone(pytz.timezone("US/Eastern")).replace(tzinfo=None)
        _t,_to=time_diff(t)
        dt.append(_t)
        to.append(_to)
        cit.append(entry['ADS']['Citations'])
dt=np.array(dt); cit=np.array(cit); to=np.array(to)
ii,=np.where(cit >=0); cit=cit[ii]; dt=dt[ii]; to=to[ii]; ii_cp=ii


n, bins, patches = plt.hist(dt, 200, facecolor='green',alpha=0.75) #makeing an histogram
plt.xlabel('Time from submission cutoff (seconds)')
plt.ylabel('Number of submissions')
plt.ylim(.1,1500)


idt=np.zeros_like(dt,dtype=int); grp=np.zeros_like(dt,dtype=int); idt_inv=np.zeros_like(dt,dtype=int)
ll=np.zeros(len(np.unique(to)),dtype=int); cit_av=np.zeros_like(ll,dtype=int)
for j,i in enumerate(sorted(np.unique(to),reverse=True)): #grouping citation per day
    ii,=np.where(to == i)
    idt[ii]=dt[ii].argsort() #submission rank, that is 1 i the submission nearest to the cutoff time, and so on
    idt_inv[ii]=idt[ii][::-1] #inverse submission rank
    ll[j]=len(ii); cit_av[j]=np.mean(cit[ii[idt[ii]][:10]])
    grp[ii] = j #group of citations submitted the same calendar day



# plt.cla()

## Kernel density estimation via Guassian kernel
ii,=np.where((idt<=5) & (cit<150)  & (cit>=2)); ii2,=np.where((idt>=10) & (idt<=20) & (cit<150)  & (cit>=2)); ii3,=np.where((idt_inv<=9) & (cit<150) & (cit>=2));
from scipy.stats.kde import gaussian_kde
x = np.linspace(0,150,20)
plt.clf()
plt.yscale('log')
lab=['1-5','5-10','last 5']
kde=gaussian_kde(cit[ii]); kde.set_bandwidth(bw_method=kde.factor *1.5 )
plt.plot(x,kde(x),label=lab[0])
kde=gaussian_kde(cit[ii2]); kde.set_bandwidth(bw_method=kde.factor *1.5 )
plt.plot(x,kde(x),label=lab[1])
kde=gaussian_kde(cit[ii3]); kde.set_bandwidth(bw_method=kde.factor *1.5 )
plt.plot(x,kde(x),label=lab[2])
plt.legend()
plt.xlabel('Submission rank')
plt.ylabel('Kernel density estimation')



#### relation between citations and submission time via hyrarchical Bayesian model


data=pd.DataFrame(np.c_[grp,grp],columns=['citations','citations_code'])
data['rank']=np.log10(idt+1)
data['log_citations']=np.log10(cit+1)
data.sort_values(by=['citations_code', 'rank'],inplace=True)
data.index=sorted(data.index)

data['log_citations'] = data['log_citations'].astype(theano.config.floatX)
citations_names = data.citations.unique()
citations_idx = data.citations_code.values

n_citations = len(data.citations.unique())

data[['citations', 'log_citations', 'rank']].head()


with pm.Model() as unpooled_model:

    # Independent parameters for each citations
    a = pm.Normal('a', 0, sd=100, shape=n_citations)
    b = pm.Normal('b', 0, sd=100, shape=n_citations)

    # Model error
    eps = pm.HalfCauchy('eps', 5)
    citations_est = a[citations_idx] + b[citations_idx]*data["rank"].values

    # Data likelihood
    y = pm.Normal('y', citations_est, sd=eps, observed=data.log_citations)


with unpooled_model:
    unpooled_trace = pm.sample(1000)




with pm.Model() as hierarchical_model:
    # Hyperpriors for group nodes
    mu_a = pm.Normal('mu_a', mu=0., sd=100**2)
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    mu_b = pm.Normal('mu_b', mu=0., sd=100**2)
    sigma_b = pm.HalfCauchy('sigma_b', 5)

    a = pm.StudentT('a', mu=mu_a, lam=sigma_a, nu=1,shape=n_citations) #intercept
    b = pm.StudentT('b', mu=mu_b, lam=sigma_b,nu=1, shape=n_citations) #slope

    # Model error
    eps = pm.HalfCauchy('eps', 5)

    citations_est = a[citations_idx] + b[citations_idx] * data["floor"].values

    # Data likelihood
    citations_like = pm.Normal('citations_like', mu=citations_est, sd=eps, observed=data.log_citations)


# Inference
with hierarchical_model:
    hierarchical_trace = pm.sample(draws=2000, n_init=1000)




pm.traceplot(hierarchical_trace);



selection = [4, 7,12,19]
fig, axis = plt.subplots(2,2, figsize=(12, 6), sharey=True, sharex=True)
axis = axis.ravel()
for i, c in enumerate(selection):
    c_data = data.ix[data.citations == c]
    c_data = c_data.reset_index(drop = True)
    c_index = np.where(citations_names==c)[0][0]
    z = list(c_data['citations_code'])[0]

    xvals = np.linspace(0, 2.5)
    for a_val, b_val in zip(unpooled_trace['a'][1000:, c_index], unpooled_trace['b'][1000:, c_index]):
        axis[i].plot(xvals, a_val + b_val * xvals, 'b', alpha=.1)
    axis[i].plot(xvals, unpooled_trace['a'][1000:, c_index].mean() + unpooled_trace['b'][1000:, c_index].mean() * xvals,
                 'b', alpha=1, lw=2., label='individual')
    for a_val, b_val in zip(hierarchical_trace['a'][1000:][z], hierarchical_trace['b'][1000:][z]):
        axis[i].plot(xvals, a_val + b_val * xvals, 'g', alpha=.1)
    axis[i].plot(xvals, hierarchical_trace['a'][1000:][z].mean() + hierarchical_trace['b'][1000:][z].mean() * xvals,
                 'g', alpha=1, lw=2., label='hierarchical')
    axis[i].scatter(c_data.floor, c_data.log_citations,
                    alpha=1, color='k', marker='.', s=80, label='original data')
    axis[i].set_title(c)


hier_a = hierarchical_trace['a'][500:].mean(axis=0)
hier_b = hierarchical_trace['b'][500:].mean(axis=0)
indv_a = [unpooled_trace['a'][500:, np.where(citations_names==c)[0][0]].mean() for c in citations_names]
indv_b = [unpooled_trace['b'][500:, np.where(citations_names==c)[0][0]].mean() for c in citations_names]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, xlabel='Intercept', ylabel='Floor Measure',
                     title='Hierarchical vs. Non-hierarchical Bayes',
                     xlim=(.5, 2.5), ylim=(-1.3, 1.3))

ax.scatter(indv_a, indv_b, s=26, alpha=0.4, label = 'non-hierarchical')
ax.scatter(hier_a,hier_b, c='red', s=26, alpha=0.4, label = 'hierarchical')
for i in range(len(indv_b)):
    ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i],
             fc="k", ec="k", length_includes_head=True, alpha=0.4, head_width=.04)
ax.legend();





ll=np.array([sum(grp ==i) for i in np.unique(grp)])
plt.plot(ll,hier_b,'k.')



weekday=np.array([datetime.datetime.fromordinal(i).isoweekday() for i in np.unique(to)])

months=np.array([datetime.datetime.fromordinal(i).month for i in np.unique(to)])
cit_hier_month=np.array([(hier_a[np.where(months ==i)[0]]+hier_b[np.where(months ==i)[0]]*np.log10(5+1)).mean()  for i in np.unique(months)])
weekdays=np.array([datetime.datetime.fromordinal(i).isoweekday() for i in to])
# cit_week=[np.mean(cit[np.where((weekdays ==i) & (cit>=0) & (cit<=60))[0]]) for i in range(1,6)]
# plt.plot(range(1,6),cit_week,'k.')


slope=np.array([np.mean(hier_b[np.where(weekday ==i)[0]]) for i in range(1,6)])
norm=np.array([np.mean(hier_a[np.where(weekday ==i)[0]]) for i in range(1,6)])
cit_hier=np.array([(hier_a[np.where(weekday ==i)[0]]+hier_b[np.where(weekday ==i)[0]]*np.log10(5+1)).mean()  for i in range(1,6)])
plt.plot(range(1,6),cit_hier,'k.')
plt.plot(range(1,6),slope,'k.')
plt.plot(range(1,6),norm,'k.')


slope=np.array([np.mean(hier_b[np.where(months ==i)[0]]) for i in range(1,13)])
norm=np.array([np.mean(hier_a[np.where(months ==i)[0]]) for i in range(1,13)])
cit_hier=np.array([(hier_a[np.where(months ==i)[0]]+hier_b[np.where(months ==i)[0]]*np.log10(5+1)).mean()  for i in range(1,13)])
plt.plot(range(1,13),cit_hier,'k.')















import pymc3 as pm
import theano
# ii=idt.argsort()
# ii,=np.where((grp==10) & (cit>=1) & (cit<=60))
x=np.log10(idt+1); y=np.log10(cit+1); x=x/max(x)
# ii,=np.where((cit>=2) & (cit<=60)); x=x[ii]; y=y[ii]
data = dict(x=x, y=y)
# data=data.ix[:20]
# data = dict(x=data['rank'].values,y=data['log_citations'].values)


with pm.Model() as model:
    pm.glm.glm('y ~ x', data)
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace = pm.sample(100, step, progressbar=True)


from theano.scan_module.scan_perform_ext import *




plt.subplot(111, xlabel='x', ylabel='y',
            title='Posterior predictive regression lines')
plt.plot(x,y, 'x', label='data')
pm.glm.plot_posterior_predictive(trace, samples=100,
                                 label='posterior predictive regression lines')

plt.legend(loc=0);


plt.hist(trace['x'])
plt.hist(trace['x'][30:])
trace['x'][30:].mean()





with pm.Model() as model_robust:
    family = pm.glm.families.StudentT()
    pm.glm.glm('y ~ x', data, family=family)
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_robust = pm.sample(200, step, progressbar=False)

plt.figure(figsize=(5, 5))
plt.plot(x,y, 'x', label='data')
pm.glm.plot_posterior_predictive(trace_robust,
                                 label='posterior predictive regression lines')
plt.legend();




plt.hist(trace_robust['x'])
plt.hist(trace_robust['x'][30:])
trace_robust['x'][60:].mean()










#### relation between citations and submission time via hyrarchical Bayesian model



##  NLP of scientific papers

titles=[];abstracts=[]
for feed in obj:
    for entry in feed.entries:
        titles.append(entry['title'])
        abstracts.append(entry['summary'])
ii=ii_cp
abstracts=np.array(abstracts)


from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
en_nlp = spacy.load('en')  # load spacy's English language models

# create a custom tokenizer using the SpaCy document processing pipeline, but with custom tokenizer
def custom_tokenizer(document):
    doc_spacy = en_nlp(document,entity=False, parse=False)
    return [token.lemma_ for token in doc_spacy]





import string
#text_train defines a list of abstracts
text_train=np.array([
    ' '.join([word for word in entry['summary'].split() if not any(x in word for x in ['$', '\\','\n','-','+']) and word.isalpha()]) #parsing the abstract
        .encode("ascii", "ignore") #removing non-ascii characters
        .lower().translate(None, string.punctuation)  #strip punctuation from a string in Python
    for feed in obj for entry in feed.entries])  # loop in entries and feeds
[ii] #ii,=np.where(cit >=0)

#title_train defines a list of titles
title_train=np.array([' '.join([word for word in entry['title'].split() if not any(x in word for x in ['$', '\\','\n','-','+']) and word.isalpha()]).encode("ascii", "ignore").lower().translate(None, string.punctuation) for feed in obj for entry in feed.entries])[ii]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), token_pattern=r'\b\w+\b', smooth_idf=False, tokenizer=custom_tokenizer).fit(text_train)  #tf_idf vectorizer with the custom tokenizer
X_train = vectorizer.fit_transform(text_train) #tf_idf matrix
feature_names = vectorizer.get_feature_names() #feature names
vectorizer2 = TfidfVectorizer(stop_words="english", ngram_range=(2, 2), token_pattern=r'\b\w+\b', smooth_idf=False, tokenizer=custom_tokenizer).fit(text_train)  #tf_idf vectorizer with the custom tokenizer
X_train2 = vectorizer2.fit_transform(text_train)
feature_names2 = vectorizer2.get_feature_names()

y_train=np.log(cit[:5000]+1)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [1e-5,5e-5,1e-4,5e-4,1e-3]}
grid = GridSearchCV(Lasso(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
feature_names = vectorizer.get_feature_names()
print("First 50 features:\n{}".format(feature_names[:50]))

from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(Lasso(alpha=1e-3),threshold=0.05).fit(X_train, y_train)
mo=Lasso(alpha=1e-4).fit(X_train, y_train); mo.score(X_train, y_train)
X_train_l1 = select.transform(X_train)
mask = select.get_support()
X_test_l1 = select.transform(X_train)
score = Lasso(alpha=1e-4).fit(X_train_l1, y_train).score(X_train_l1, y_train)
mo=Lasso(alpha=1e-4).fit(X_train_l1, y_train)
mo.coef_


max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
print("Features with highest tfidf: \n{}".format(feature_names[sorted_by_tfidf[-100:]]))


from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=20)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)
# select2=SelectPercentile(percentile=20)
# X_train_selected2 = select2.fit(X_train2,y_train).transform(X_train2)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))


mask = select.get_support()
print(mask)
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

aa=np.where(X_train_selected.sum(0) >50)[1]
feat=np.array(feature_names)[mask][aa]
import nltk
pos_tagged_tokens = [nltk.pos_tag([t]) for t in feat]
pos_tagged_tokens = [token for sent in pos_tagged_tokens for token in sent]

current_entity_chunk=[]
for (token, pos) in pos_tagged_tokens:
        if pos.startswith('JJ'):
            current_entity_chunk.append(token)


np.array(feature_names)[mask][np.where(X_train_selected.sum(0) >50)[1]]
# np.array(feature_names2)[select2.get_support()][np.where(X_train_selected2.sum(0) >50)[1]]

ii2=(X_train[:,feature_names == 'issue'] >0).toarray().flatten() + (X_train[:,feature_names == 'disagreement'] >0).toarray().flatten()+ (X_train[:,feature_names == 'tension'] >0).toarray().flatten() + (X_train[:,feature_names == 'contrast'] >0).toarray().flatten() #+ (X_train[:,feature_names == 'disprove'] >0).toarray().flatten()
_=plt.hist(cit[ii2], 30, facecolor='red',alpha=0.75,normed=1)
ii2=(X_train[:,feature_names == 'agreement'] >0).toarray().flatten() + (X_train[:,feature_names == 'confirm'] >0).toarray().flatten() +(X_train[:,feature_names == 'validate'] >0).toarray().flatten()
_=plt.hist(cit[ii2], 30, facecolor='yellow',alpha=0.75,normed=1)
plt.yscale('log');plt.xscale('log')


from scipy.stats.kde import gaussian_kde
x = np.linspace(0,200,20)
plt.clf()
plt.yscale('log');plt.xscale('log')
feature_names=np.array(feature_names)
linestyles = ['-','-','-','-','--', '--','--',':',':']
for k,i in enumerate(['unprecedented','exceptionally','significantly','strongly','speculate','suggest','indicate','contrast','agreement']):
    plt.plot(x,gaussian_kde(cit[(X_train[:,feature_names == i] >0).toarray().flatten()])(x),linestyles[k],label=i)
# for i in ['good agreement']:
#     plt.plot(x, gaussian_kde(cit[(X_train2[:, feature_names2 == i] > 0).toarray().flatten()])(x), label=i)
plt.ylim([1e-5,0.08])
plt.legend(loc="best")
plt.xlabel('Citation'); plt.ylabel('KDE')


feature_names2=np.array(feature_names2)
for i in ['good agreement','previous work','provide good','result indicate','strongly indicate','strong indication', '-PRON- demonstrate','-PRON- prove']:
    plt.plot(x,gaussian_kde(cit[(X_train2[:,feature_names2 == i] >0).toarray().flatten()
                            ])(x),label=i)
plt.ylim([1e-4,3e-2])
plt.legend(loc="best")
plt.yscale('log');plt.xscale('log')



x = np.linspace(0,800,60)
ii2=(X_train[:,feature_names == 'issue'] >0).toarray().flatten() + (X_train[:,feature_names == 'disagreement'] >0).toarray().flatten()+ (X_train[:,feature_names == 'tension'] >0).toarray().flatten() + (X_train[:,feature_names == 'contrast'] >0).toarray().flatten() + (X_train[:,feature_names == 'disprove'] >0).toarray().flatten()
kde=gaussian_kde(cit[ii2]); kde.set_bandwidth(bw_method=kde.factor *5 )
plt.plot(x,kde(x),label=':)')
ii2=(X_train[:,feature_names == 'agree'] >0).toarray().flatten() + (X_train[:,feature_names == 'confirm'] >0).toarray().flatten() +(X_train[:,feature_names == 'validate'] >0).toarray().flatten()
kde=gaussian_kde(cit[ii2]); kde.set_bandwidth(bw_method=kde.factor * 10)
plt.plot(x,kde(x),label=':(')
plt.yscale('log');plt.xscale('log')
plt.ylim([1e-6,0.05]); plt.xlim(10,800)
plt.legend(loc="best")







#Analysis of citation index

publ=[];tdiff=[]
for feed in obj:
    for entry in feed.entries:
        comment=entry['arxiv_comment'].lower() if entry.has_key('arxiv_comment') else ''
        if comment:
            if any(x in comment for x in ['accept', 'press', 'publish']):
                if 'v1' in entry['id']:
                    _publ = 2 ; _tdiff=0
                else:
                    _publ=1
                    _tdiff= (parse(entry['updated'])-parse(entry['published'])).total_seconds()/(24*3600.*30)
        else:
            _publ=-1; _tdiff=-1

        publ.append(_publ)
        tdiff.append(_tdiff)

ii=ii_cp;publ=np.array(publ)[ii]
plt.xscale('log');plt.yscale('log')
kde=gaussian_kde(cit[np.where(publ==2)[0]]); kde.set_bandwidth(bw_method=kde.factor *3 )
plt.plot(x,kde(x),label='In press at v1')
kde=gaussian_kde(cit[np.where(publ==1)[0]]); kde.set_bandwidth(bw_method=kde.factor *3 )
plt.plot(x,kde(x),label='Submitted at v1')
plt.legend(loc="best")
plt.ylabel('KDE'); plt.xlabel('Citations')



# number of authors vs. citaton
num_authors=[]
for feed in obj:
    for entry in feed.entries:
        num_authors.append(len(entry['authors']) if entry.has_key('authors') else 0)
ii=ii_cp
num_authors=np.array(num_authors)[ii]
plt.plot(num_authors,cit,'k.')
plt.yscale('log')





#self promoted SF vs. visibility bias VB
x = np.linspace(0,200,20)
kde=gaussian_kde(cit[np.where((idt<=8)  & (cit>=1) & (cit<=400) & (dt<600))]); kde.set_bandwidth(bw_method=kde.factor *3 )
plt.plot(x,kde(x),label='SP')
kde=gaussian_kde(cit[np.where((idt<=8)  & (cit>=1) & (cit<=400) & (dt>=600))]); kde.set_bandwidth(bw_method=kde.factor *3 )
plt.plot(x,kde(x),label='VB')
kde=gaussian_kde(cit[np.where((idt>=20)  & (cit<=400) & (cit>=1))]); kde.set_bandwidth(bw_method=kde.factor *3 )
plt.plot(x,kde(x),label='others')
plt.yscale('log');plt.xscale('log')
plt.legend(loc="best")


import re
journals=[]
for feed in obj:
    for entry in feed.entries:
        _journal=re.findall(r'([a-zA-Z&]{2,})',entry['ADS']['Bibliographic Code'])[0] \
            if entry['ADS']['Bibliographic Code'] and type(entry['ADS']['Bibliographic Code'])!=list else ''
        if _journal =='ApJ' and re.findall(r'(ApJ\S{4,}L)',entry['ADS']['Bibliographic Code']): _journal='ApJL'
        journals.append(_journal)
ii=ii_cp
journals=np.array(journals)[ii]

imp_factor=[4.952,5.909,5.487,11.257,5.185,38.138,34.661]  #refer to 2015
for k,i in enumerate(['MNRAS','ApJ','ApJL','ApJS','A&A','Natur','Sci']):
    kde = gaussian_kde(cit[(journals == i) & (cit>=20)])
    width=1 if i in ['Natur','Sci'] else 1.8
    if i in ['ApJ', 'ApJL','MNRAS']: width = 4
    kde.set_bandwidth(bw_method=kde.factor * width)
    plt.plot(x,kde(x),label=i+' '+str(imp_factor[k]))
# plt.ylim([1e-5,6e-2])
plt.legend(loc="best")
plt.yscale('log');plt.xscale('log')



import urllib2
html = urllib2.urlopen('http://adsabs.harvard.edu/abs_doc/refereed.html').read()  # ADS Bibliographic Codes: Refereed Publications
biblio_code=[i.strip() for i in re.findall('onClick=.*>(.*)<',html)]; biblio_code.append('ApJL')
journals=[re.sub(r'[\d.]+','',''.join(\
    re.findall(r'([a-zA-Z&]{2,})|\d+(ApJ[.\d]+L)',entry['ADS']['Bibliographic Code'])[0])) \
              if entry['ADS']['Bibliographic Code'] and type(entry['ADS']['Bibliographic Code'])!=list \
              else '' \
          for feed in obj for entry in feed.entries]
biblio_code=tuple(set(journals).intersection(biblio_code))


# Impact_factor
# https://en.wikipedia.org/wiki/Impact_factor#cite_note-IFintro-1
import pandas as pd, re
df0= pd.DataFrame(index=np.arange(1980,2019),columns=np.unique(journals)) #the total number of articles published in a journal in a specific month
df= df0.copy() # the number of citations received in a specific month by articles published in a journal
# _t_pub0=datetime.datetime(2013,1,1)
for feed in obj:
    for entry in feed.entries:
        # if entry['ADS']['Citations'] <100: continue
        _journal=re.findall(r'([a-zA-Z&]{2,})',entry['ADS']['Bibliographic Code'])[0] \
            if entry['ADS']['Bibliographic Code'] and type(entry['ADS']['Bibliographic Code'])!=list else ''
        if _journal == 'ApJ' and re.findall(r'(ApJ\S{4,}L)', entry['ADS']['Bibliographic Code']): _journal = 'ApJL'
        try:
            # _t_pub0=datetime.datetime.strptime(entry['ADS']['Publication Date'].replace('00/','01/'), '%m/%Y').year
            _t_pub0=int(entry['ADS']['Publication Date'].split('/')[1])
        except:
            continue
        df0[_journal].ix[_t_pub0+1:_t_pub0 + 2] = df0[_journal].ix[_t_pub0+1:_t_pub0 + 2].fillna(0) + 1

        for _t in entry['ADS']['Citations list']:
            __journal= re.findall(r'([a-zA-Z&]{2,})', _t[0])[0] \
            if entry['ADS']['Bibliographic Code'] else ''
            if __journal == 'ApJ' and re.findall(r'(ApJ\S{4,}L)', entry['ADS']['Bibliographic Code']): __journal = 'ApJL'
            try:
                # _t_pub=datetime.datetime.strptime(_t[2].replace('00/','01/'), '%m/%Y').year
                _t_pub = int(_t[2].split('/')[1])
            except:
                continue
            if __journal in biblio_code:
                if _t_pub-_t_pub0==0:
                    df[_journal].ix[_t_pub0+1:_t_pub0+2]=df[_journal].ix[_t_pub0+1:_t_pub0+2].fillna(0) + 1
                elif _t_pub-_t_pub0==1:
                    df[_journal].ix[_t_pub0 + 2] = df[_journal].ix[_t_pub0 + 2]+ 1 if pd.notnull(df[_journal].ix[_t_pub0+2]) else 1

df_cp=df.copy();df0_cp=df0.copy()
df0=df0.loc[df.notnull().any(axis=1)]
df=df.loc[df.notnull().any(axis=1)]
# df0.loc['2001-01-01':'2016-12-31',:] = df0.loc[_t_pub0].values/(48.)


i=['MNRAS','ApJ','ApJL','ApJS','A&A','Natur','Sci']
pd.DataFrame(df[i]/df0[i]).plot()

pd.DataFrame(df[i].rolling(window=24,min_periods=12).sum()/df0[i].rolling(window=24,min_periods=12).sum().values,index=df0.index.shift(0)).plot()

pd.DataFrame((df[i].rolling(window=24,min_periods=12).sum()/df0[i].rolling(window=24,min_periods=12).sum()).values,index=df0.index.shift(24,freq='M'),columns=i).plot()

# df0.where(df >=40,np.nan,inplace=True); df.where(df >=100,np.nan,inplace=True)
# df0=df0_cp.copy(); df=df_cp.copy()















def get_journal(journal):
    _journal=re.findall(r'([a-zA-Z&]{2,})', journal)[0] \
        if journal and type(journal) != list else ''
    if _journal == 'ApJ' and re.findall(r'(ApJ\S{4,}L)', journal): _journal = 'ApJL'
    return _journal

import pandas as pd, re
jou=['MNRAS','ApJ','ApJL','ApJS','A&A','Natur','Sci']
df=pd.Panel(items =[0,1], major_axis = np.arange(2014,2017), minor_axis = jou) #https://stackoverflow.com/questions/23431900/fill-pandas-panel-object-with-data
for feed in obj:
    for entry in feed.entries:
        # if entry['ADS']['Citations'] <100: continue
        _journal=get_journal(entry['ADS']['Bibliographic Code'])
        try:
            _t_pub0=int(entry['ADS']['Publication Date'].split('/')[1])
            if _journal not in df[0].keys() or _t_pub0 not in [2013,2014,2015]: continue
        except:
            continue
        df[0][_journal].ix[_t_pub0+1:_t_pub0+2]=df[0][_journal].ix[_t_pub0+1:_t_pub0+2].fillna(0)+1

        for _t in entry['ADS']['Citations list']:
            __journal=get_journal(_t[0])
            try:
                _t_pub = int(_t[2].split('/')[1])
            except:
                continue
            if __journal in biblio_code and _t_pub in [2015,2016] and 1<=_t_pub-_t_pub0<=2:
                df[1][_journal].ix[_t_pub]=df[1][_journal].ix[_t_pub]+1 if pd.notnull(df[1][_journal].ix[_t_pub]) else 1



df_cp=df.copy()
(df[1]/df[0]).ix[2015:2016].plot(kind='bar',rot=0)

(df[1]/df[0]).ix[2015] / [4.952,5.909,5.487,11.257,5.185,38.138,34.661]












bins=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,40,50,60,70,80,100,130,160,200,250])
bins_center=(bins[:-1] + bins[1:]) / 2.; bins_center=np.append(bins_center,250)
df3=pd.DataFrame(index=bins_center, columns = jou) #KDE IF 2015
for feed in obj:
    for entry in feed.entries:
        _journal=get_journal(entry['ADS']['Bibliographic Code'])
        try:
            _t_pub0=int(entry['ADS']['Publication Date'].split('/')[1])
            if _journal not in df[0].keys() or _t_pub0 not in [2013,2014]: continue
        except:
            continue

        cit_thresh=0
        for _t in entry['ADS']['Citations list']:
            __journal=get_journal(_t[0])
            try:
                _t_pub = int(_t[2].split('/')[1])
            except:
                continue
            if __journal in biblio_code and _t_pub == 2015: cit_thresh+=1
        cit_thresh=bins_center[np.digitize(cit_thresh, bins=bins)-1]
        df3[_journal].ix[cit_thresh]=df3[_journal].ix[cit_thresh]+1 if pd.notnull(df3[_journal].ix[cit_thresh]) else 1


df3.rolling(window=10,center=True,min_periods=1).mean().div(df3.sum(0)).plot(logy=True,logx=True)
plt.xlim([1,250])



