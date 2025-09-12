import configparser
import requests

class SportslineData:
    '''Class to download Sportsline articles into S3.

    Parameters:
        sport: 'nascar', 'nfl', 'nba', 'mlb', 'nhl', or 'pga'
        
    '''
    #load config parameters for sportsline
    cfg = configparser.ConfigParser()
    cfg.read('./configs/etl.ini')
    user_id = cfg['CREDENTIALS']['user_id']
    password = cfg['CREDENTIALS']['password']

    #load login details for sportsline
    login = 'https://secure.sportsline.com/login'
    payload = {
        'dummy::login_form': '1',
        'form::login_form': 'login_form',
        'xurl': 'http://secure.sportsline.com/',
        'master_product': '23350',
        'vendor': 'sportsline',
        'form_location': 'log_in_page',
        'userid': user_id,
        'password': password
    }
   
   
#    def __init__(self, sport):
#        self.sport = sport if sport != 'pga' else 'golf' #confined to nhl, nascar, nfl, pga, nba, mlb
#        self.articles = 'https://www.sportsline.com/sportsline-web/service/v1/articleIndexContent?slug={}&limit=10000&auth=1'.format(self.sport)
#        self.url = 'https://www.sportsline.com/sportsline-web/service/v1/articles/{}?auth=1'
#        self.folder = '{}/sportsline'.format(sport)
   
   
    def fetch_and_save_raw_html(url, file_path):
        try:
            # Make a GET request to fetch the raw HTML content
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            
            # Get the raw HTML content
            raw_html = response.text
            
            # Save the raw HTML content to a file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_html)
            
            print(f"Raw HTML content successfully saved to {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")


#    def _get_links(self):
#        '''Get relevant article links to download.'''
#        page = json.loads(requests.get(self.articles).content.decode())
#        links = []
       
#        for article in page['articles']:
#            title = article['slug']
           
#            if self.sport == 'nascar':
#                if title.endswith('from-a-dfs-pro') or title.startswith('nascar-at') or title.startswith('projected-nascar-leaderboard'):
#                    links.append(article['slug'])

#            elif self.sport == 'golf':
#                if title.endswith('from-a-dfs-pro') or title.endswith('has-surprising-picks-and-predictions'):
#                    links.append(article['slug'])
                   
#        return links
   
   
#    def update_articles(self):

#        links = self._get_links()
#        bucket = self.s3.Bucket(self.bucket_name)

#        for obj in bucket.objects.all():
#            if self.sport != 'golf':
#                if 'sportsline' in obj.key and self.sport in obj.key:
#                    key = obj.key.split('/')[-1]
#                    if key in links:
#                        links.remove(key)
#            else:
#                if 'sportsline' in obj.key and 'pga' in obj.key:
#                    key = obj.key.split('/')[-1]
#                    if key in links:
#                        links.remove(key)
               
#        #Open session and post the user_id and password
#        with requests.Session() as session:
#            post = session.post(self.login, data=self.payload)

#            #open links
#            for link in links:
#                article = self.url.format(link)
#                page = session.get(article)
#                page = json.loads(page.text)

#                #save article into s3 object
#                obj = self.s3.Object(self.bucket_name, '{}/{}'.format(self.folder, link))
#                obj.put(
#                    Body=json.dumps(page['article'])
#                )
   
#                print(link) 
   

