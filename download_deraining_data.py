from google_drive_downloader import GoogleDriveDownloader as gdd
import os

gdd.download_file_from_google_drive(file_id='17ZvG1YRzJwMDdFCmi5otKOo2CQfMqf1L',
                                    dest_path='./data3.zip',
                                    unzip=True)


gdd.download_file_from_google_drive(file_id='1VGefOgfL62BPnANR3jWUygZ9nq-rbfyO',
                                    dest_path='./data4.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='1tmSJXHmZH7DBkPRP_iTq0P5wtUYIEfPw',
                                    dest_path='./data5.zip',
                                    unzip=True)

os.system('rm -rf RainData/Rain100H')
os.system('rm -rf RainData/RainTrainH')
