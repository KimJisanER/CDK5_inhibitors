import os
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import messagebox

#file 변수에 선택 파일 경로 넣기
file1 = filedialog.askopenfilenames(initialdir="/",\
                 title = "파일을 선택 해 주세요",\
                    filetypes = (("*.csv","*csv"),("*.xlsx","*xlsx"),("*.xls","*xls")))

#파일 선택 안했을 때 메세지 출력

if file1 == '':
    messagebox.showwarning("경고", "파일을 추가 하세요")

print(file1)    #files 리스트 값 출력

#dir_path에 파일경로 넣어서 읽기
for dir_path in file1:
    prior = pd.read_csv(dir_path ,sep=',', encoding='cp949')

print(len(prior))


def candidate(data):
  c = []
  for i in range(len(data)):
    c.append(data['uid'][i])
  return(c)

#리스트 비교
download_list=candidate(prior)

#출력
print()
print('zinc_list:\n',download_list,'\n')

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.chrome.service import Service

driver = webdriver.Chrome('./chromedriver')

try:

    driver.get("https://zinc15.docking.org/subclasses/kinase/substances/")

    for i in range(len(download_list)):
        # 검색
        elem = driver.find_element_by_xpath('/html/body/div/div/div/nav[1]/div/div/div[4]/div[3]/form/input')
        ac = ActionChains(driver)
        ac.move_to_element(elem)
        ac.click()
        ac.send_keys(download_list[i])
        # elem.submit()
        ac.perform()
        elem.send_keys(Keys.RETURN)

        time.sleep(2)
        try:
            # 첫번째 항목 누르기
              driver.find_elements_by_css_selector('#print > div > div > div > h4 > a')[0].click()
#             # 다운로드 누르기
              driver.find_elements_by_css_selector('body > div > div > div > div:nth-child(2) > div.col-sm-9 > div:nth-child(3) > table:nth-child(1) > tbody > tr > td:nth-child(6) > div > button')[0].click()
# #             # smi고르기
#               driver.find_elements_by_css_selector('body > div > div > div > div:nth-child(2) > div.col-sm-9 > div:nth-child(3) > table:nth-child(1) > tbody > tr > td:nth-child(6) > div > ul > li:nth-child(1)')[0].click()
            # sdf고르기
              driver.find_elements_by_css_selector('body > div > div > div > div:nth-child(2) > div.col-sm-9 > div:nth-child(3) > table:nth-child(1) > tbody > tr > td:nth-child(6) > div > ul > li:nth-child(2)')[0].click()
#             # 목록으로 복귀
              driver.get("https://zinc15.docking.org/subclasses/kinase/substances/")
#
        except Exception as e:
            print('오류 : ',download_list[i])
            print(e)
            driver.get("https://zinc15.docking.org/subclasses/kinase/substances/")
#
except Exception as e:
    print('오류발생')
    print(e)