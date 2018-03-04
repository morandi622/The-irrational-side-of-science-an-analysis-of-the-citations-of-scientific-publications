
"""
Filename: ArXivRunner.py
Author: Andrea Morandi
e-mail: andrea.morandi@uah.edu

Purpose: submit a paper on astro-ph at 14.00 EDT **sharp** to have the paper appear at the top on the general “new preprints” list

Relevant links (the cutooff time may be outdated):
https://arxiv.org/help/submit#availability
http://faculty.virginia.edu/petrov//blog/2014/01/10/what-time-of-the-day-submit-to-arXiv/

Remember to sincronize your pc clock before submission

"""


# tested on python 2.7
import mechanize, pytz  #pip install mechanize
from bs4 import BeautifulSoup
from datetime import datetime
from threading import Timer

######  parameters to be changed  ############
time=datetime(2017,5,1,14,0,0,0)  #desired time EDT of submission, e.g. May, 1 2017 14.00 EDT (year, month, day, hour, minute, second, microsecond
url='https://arxiv.org/submit/1877014/preview'  #url of your paper to be submitted on astro-ph
username='andrea.morandi@uah.edu' #username for astro-ph
password='your_password'   #password for astro-ph
###########################################


# automatic timezone conversion of the submission time: EDT to local timezone
time=(pytz.timezone("US/Eastern").localize(time).astimezone(pytz.utc).replace(tzinfo=None) - (datetime.utcnow() - datetime.now()))


br = mechanize.Browser()
br.set_handle_robots(False)
resp=br.open(url)
br.select_form(nr = 1)  #selecting the form for login
br.form['username'] = username  #username for astro-ph
br.form['password'] = password   #password
response=br.submit()  #logging in

# cross-check below if you logged in
userPage = BeautifulSoup(response, "lxml") #parsing the user page
userPage.find(attrs={'class': 'title'}).get_text() #fecthing the title
userPage.find(attrs={'class': 'authors'}).get_text() #fecthing the authors
userPage.find(attrs={'class': 'abstract'}).get_text() #fetching the abstract

# function for selecting the submission form
def select_form(form):
  return form.attrs.get('action', None) == url.replace('preview','submit')

br.select_form(predicate=select_form)  #selecting the submission button
br.form.attrs  #cross-check  if you got the Submit button

#function to click the submssion button
def submit_astroph():
    br.submit()

#time scheduler
Timer((time-datetime.today()).seconds, submit_astroph).start()


