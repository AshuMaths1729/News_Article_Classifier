# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:09:19 2018

@author: Ashutosh
"""

import scrapy
import pandas as pd
from datetime import timedelta, date
import datetime

class MyItem(scrapy.Item):
	Title = scrapy.Field()
	Date = scrapy.Field()
	article = scrapy.Field()

def daterange(start_date, end_date):
	for n in range(int((end_date - start_date).days)):
		yield start_date + timedelta(n)

class ToiSpider(scrapy.Spider):
	name = 'toi'
	allowed_domains = ['timesofindia.indiatimes.com']
	keyword = "HIV"

	def start_requests(self):
		
		min_date = datetime.datetime.strptime('1/1/2010', '%d/%m/%Y')
		max_day = (date.today()).strftime('%d/%m/%Y')
		max_date = datetime.datetime.strptime(max_day, '%d/%m/%Y')
		start_date = datetime.datetime.strptime('1/1/2010', '%d/%m/%Y')
		end_date = datetime.datetime.strptime('4/12/2018', '%d/%m/%Y')	
		start = pd.datetime(2010,1,1).date()
		end = date.today()
		N = int((end-start).days+1)
		data = pd.DataFrame({'A': range(40179, 40179 + N)}, index=pd.date_range(start=start, end=end, freq='D'))
		start_date = start_date.strftime('%Y-%m-%d')
		end_date = end_date.strftime('%Y-%m-%d')
		start_page_no = data.ix[start_date, 'A']
		end_page_no = data.ix[end_date, 'A']
		urls = ['https://timesofindia.indiatimes.com/2010/1/1/archivelist/year-2010,month-1,starttime-%d.cms' % page for page in range(start_page_no, end_page_no + 1)]
		for url in urls:
			yield scrapy.Request(url=url, callback=self.parse_day)

	def parse_day(self, response):
		sel_article = response.xpath('//span/a[contains(text(), "'+self.keyword+'")]')
		sel_link = sel_article.xpath('@href').extract()
		
		for link in sel_link:
			yield scrapy.Request(response.urljoin(link), callback=self.parse_article)

	def parse_article(self, response):
		item = MyItem()
		item['Title'] = response.xpath('//section/h1/arttitle/text()').extract()
		item['Date'] = response.xpath('//section/span/span/text()').extract()
		if response.xpath('//arttextxml/text()'):
			item['article'] = response.xpath('//arttextxml/text()').extract()
		elif response.css('div.Normal span::text'):
			item['article'] = response.css('div.Normal span::text').extract()
		else:
			item['article'] = response.css('div.Normal::text').extract()
		yield item