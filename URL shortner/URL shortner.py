import sys
#print(sys.version)

#URL shortner
import random
import string

d = dict() #initialising master dictionary that contains all the shortUrl of longUrl

def urlShort(longUrl):
  '''
  input  : longUrl 
  output : returns short url of string type 
  https://www.shortUrl.com/"+shortUrl
  first 25 characters are common to all  
  '''
    
    l = random.randint(6,10) #random value in between 6-10
    
    chars = string.ascii_lowercase 
    shortUrl = ''.join(random.choice(chars) for i in range(l)) #generate random string of length l
    
    if shortUrl in d:
        return urlShort(longUrl)
    else:
        d[shortUrl]=longUrl
    return "https://www.shortUrl.com/"+shortUrl #https://www.. is 25 character long

def urlLong(shortUrl):
  """"
  input  : shortUrl of string type-- https://www.shortUrl.com/"+shortUrl
  output : longURL
  """
    k = shortUrl[25:]
    if k in d:
        return d[k]
    else:
        return "No such URL in our data base"
      
