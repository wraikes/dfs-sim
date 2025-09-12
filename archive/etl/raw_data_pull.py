import os
import json
import time
import requests
import glob
from datetime import datetime

from common import constants as con


class ExtractRawData:
    '''Simple class to download historical Linestarapp data. Projections must be downloaded manually.'''

    parameters = {
        con.NFL: {'sport': 1, 'pid_start': 250},
        con.NBA: {'sport': 2, 'pid_start': 304},
        con.MLB: {'sport': 3, 'pid_start': 550},
        con.PGA: {'sport': 5, 'pid_start': 234},
        con.NHL: {'sport': 6, 'pid_start': 338},
        con.MMA: {'sport': 8, 'pid_start': 237},
        con.NASCAR: {'sport': 9, 'pid_start': 209}
    }

    def __init__(self, sport):
        self.sport = sport
        self.sport_id = self.parameters[self.sport]['sport']
        self.pid_start = self.parameters[self.sport]['pid_start']
        self.site = {'dk': 1, 'fd': 2}
        self.html = 'https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV4?sport={}&site={}&periodId={}'
        self.folder = f'/home/wraikes/programming/dfs/data/{self.sport}'

    def pull_data(self, clean_projections=True):
        """Pull all available historical data. Skips projections (download those manually)."""
        
        if clean_projections:
            self._clean_old_projections()
        
        print(f"Starting historical data pull for {self.sport}")
        pid = self._get_max_pid() + 1
        print(f"Starting from pid: {pid}")

        consecutive_failures = 0
        total_files_saved = 0

        while consecutive_failures < 25:
            files_saved_this_pid = 0

            for site, site_num in self.site.items():
                if pid == 417 and self.sport == 'nascar':
                    continue  # Known bad pid for NASCAR

                try:
                    data = self._pull_json_data(pid, site_num)
                    
                    if not self._is_valid_data(data):
                        continue
                    
                    if self._is_projection(data):
                        print(f"  {site} pid {pid}: Skipping projections (download manually)")
                        continue
                    
                    # Save historical data
                    file_path = f'{self.folder}/raw/{site}_{pid}.json'
                    with open(file_path, 'w') as file:
                        json.dump(data, file)
                    
                    print(f"  {site} pid {pid}: Saved historical data")
                    files_saved_this_pid += 1
                    total_files_saved += 1

                except Exception as e:
                    continue  # Fail silently and try next site

            if files_saved_this_pid == 0:
                consecutive_failures += 1
                if consecutive_failures == 10:
                    if not self._probe_ahead(pid + 1):
                        break
            else:
                consecutive_failures = 0

            pid += 1

        print(f"Completed! Saved {total_files_saved} historical files.")
        print(f"Remember to manually download projections files when needed.")

    def _clean_old_projections(self):
        """Remove old projections files before starting."""
        projections_files = glob.glob(f'{self.folder}/raw/*projections*.json')
        
        if projections_files:
            print(f"Cleaning {len(projections_files)} old projections files...")
            for file in projections_files:
                os.remove(file)
                print(f"  Removed: {os.path.basename(file)}")
        else:
            print("No old projections files to clean.")

    def _pull_json_data(self, pid, site_num):
        """Make API request and return JSON data."""
        url = self.html.format(self.sport_id, site_num, str(pid))
        
        response = requests.get(url, timeout=30)
        time.sleep(1)  # Be nice to their API
        response.raise_for_status()
        
        return response.json()

    def _is_valid_data(self, data):
        """Check if response contains valid data."""
        return data.get('PositionFilters') is not None

    def _is_projection(self, data):
        """Check if data is projections (True) or historical (False)."""
        try:
            salary_data = json.loads(data['SalaryContainerJson'])
            
            # Check if all players have zero points scored (projections)
            players = salary_data.get('Salaries', [])
            ps_values = [p.get('PS', 0) for p in players if isinstance(p.get('PS'), (int, float))]
            
            return sum(ps_values) == 0
            
        except Exception:
            return False

    def _probe_ahead(self, start_pid, probe_range=50, probe_step=5):
        """Check if there's more data ahead by sampling future PIDs."""
        for probe_pid in range(start_pid, start_pid + probe_range + 1, probe_step):
            for site, site_num in self.site.items():
                try:
                    data = self._pull_json_data(probe_pid, site_num)
                    if self._is_valid_data(data):
                        return True
                except Exception:
                    continue
        return False

    def _get_max_pid(self):
        """Find the highest PID from existing files."""
        max_pid = self.pid_start
        raw_folder = f'{self.folder}/raw/'

        if not os.path.exists(raw_folder):
            return max_pid

        for filename in os.listdir(raw_folder):
            if 'projections' in filename:
                continue  # Skip projections files
                
            try:
                # Extract PID from filename like "dk_492.json"
                pid = int(filename.split('.')[0].split('_')[1])
                max_pid = max(max_pid, pid)
            except (IndexError, ValueError):
                continue

        return max_pid


# class SportslineData:
#    '''Class to download Sportsline articles into S3.
   
#    Parameters:
#        sport: 'nascar', 'nfl', 'nba', 'mlb', 'nhl', or 'pga'
       
#    '''
#    #load s3 bucket details
#    s3 = boto3.resource('s3')
#    bucket_name = 'my-dfs-data'
#    bucket = s3.Bucket(bucket_name)

#    #load config parameters for sportsline
#    cfg = configparser.ConfigParser()
#    cfg.read('./raw_data_pull/etl.ini')
#    user_id = cfg['CREDENTIALS']['user_id']
#    password = cfg['CREDENTIALS']['password']

#    #load login details for sportsline
#    login = 'https://secure.sportsline.com/login'
#    payload = {
#        'dummy::login_form': '1',
#        'form::login_form': 'login_form',
#        'xurl': 'http://secure.sportsline.com/',
#        'master_product': '23350',
#        'vendor': 'sportsline',
#        'form_location': 'log_in_page',
#        'userid': user_id,
#        'password': password
#    }
   
   
#    def __init__(self, sport):
#        self.sport = sport if sport != 'pga' else 'golf' #confined to nhl, nascar, nfl, pga, nba, mlb
#        self.articles = 'https://www.sportsline.com/sportsline-web/service/v1/articleIndexContent?slug={}&limit=10000&auth=1'.format(self.sport)
#        self.url = 'https://www.sportsline.com/sportsline-web/service/v1/articles/{}?auth=1'
#        self.folder = '{}/sportsline'.format(sport)
   
   
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
   

