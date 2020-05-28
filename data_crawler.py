# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:48:14 2019

@author: sidhant
"""

from icrawler.builtin import GoogleImageCrawler

#google_crawler = GoogleImageCrawler(storage={'root_dir': 'C:\sidhant\CA2\data\junkfood'})
#google_crawler.crawl(keyword='onion rings', max_num=500)
#google_crawler = GoogleImageCrawler(storage={'root_dir': 'C:\sidhant\CA2\data\junkfood'})
#google_crawler.crawl(keyword='cup cakes', max_num=5000)
#google_crawler = GoogleImageCrawler(storage={'root_dir': 'C:\sidhant\CA2\data\junkfood'})
#google_crawler.crawl(keyword='donuts', max_num=5000)
google_crawler = GoogleImageCrawler(storage={'root_dir': 'C:\sidhant\CA2\data\healthyfood'})
google_crawler.crawl(keyword='israeli salad', max_num=5000)