from google_drive_downloader import GoogleDriveDownloader as gdd
import os

gdd.download_file_from_google_drive(file_id='1njZjLsX8bWxazA3hY3TFJaBughmSglgY',
                                    dest_path='./data.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='1iymLW8Vg8MWcmrsjpJHOYCRBeQpcwh72',
                                    dest_path='./data2.rar',
                                    unzip=False)

os.system('unrar x data2.rar')
os.system('rm -rf exploration_database_and_code/data')
os.system('rm -rf exploration_database_and_code/result')
os.system('rm -rf exploration_database_and_code/support_functions')
os.system('rm -rf exploration_database_and_code/demo.m')
os.system('rm -rf exploration_database_and_code/ssim.m')