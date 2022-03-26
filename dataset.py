import requests
import csv
import os


def download_dataset(url,fileName):
    r = requests.get(url)
    # content = r.json()
    f = open(fileName,"w")
    f.write(r.text.replace("\r\n","\n"))
    f.close()


class covid_dataset():
    headers = 'case_number, report_date, date_of_onset, gender, age, patient_status, residence, confirm_status, date, confirmed_case_per_day, death_cases, discharge_cases, probable_cases, critical_condition_cases'
    case_number = list()
    report_date = list()
    date_of_onset = list()
    gender = list()
    age = list()
    patient_status = list()
    residence = list()
    classification = list()
    confirm_status = list()

    district = list()
    case_number_1 = list()
    building_name = list()

    date = list()
    confirmed_case = list()
    confirmed_case_per_day = list()
    death_cases = list()
    discharge_cases = list()
    probable_cases = list()
    critical_condition_cases = list()


    download_dataset("http://www.chp.gov.hk/files/misc/building_list_eng.csv","building_list_eng.csv")
    download_dataset("http://www.chp.gov.hk/files/misc/enhanced_sur_covid_19_eng.csv","enhanced_sur_covid_19_eng.csv")
    download_dataset("http://www.chp.gov.hk/files/misc/latest_situation_of_reported_cases_covid_19_eng.csv","latest_situation_of_reported_cases_covid_19_eng.csv")

    # os.system('wget -O building_list_eng.csv http://www.chp.gov.hk/files/misc/building_list_eng.csv')
    # os.system('wget -O enhanced_sur_covid_19_eng.csv http://www.chp.gov.hk/files/misc/enhanced_sur_covid_19_eng.csv')
    # os.system('wget -O latest_situation_of_reported_cases_covid_19_eng.csv http://www.chp.gov.hk/files/misc/latest_situation_of_reported_cases_covid_19_eng.csv')

    with open('latest_situation_of_reported_cases_covid_19_eng.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        # # 以迴圈輸出每一列
        first_row = True
        first_data = True
        cache_value = 0
        for row in rows:
            if first_row:
                first_row = False
                continue
            if not first_row:
                date.append(row[0])
                confirmed_case.append(row[2])
                if first_data == True:
                    confirmed_case_per_day.append(int(row[2]))
                    first_data = False
                elif first_data == False:
                    confirmed_case_per_day.append(int(row[2])-cache_value)
                    cache_value = int(row[2])
                death_cases.append(row[6])
                discharge_cases.append(row[7])
                probable_cases.append(row[8])
                critical_condition_cases.append(row[9])

    with open('enhanced_sur_covid_19_eng.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        # # 以迴圈輸出每一列
        for row in rows:
            if first_row:
                first_row = False
                continue
            if not first_row:
                case_number.append(row[0])
                report_date.append(row[1])
                date_of_onset.append(row[2])
                gender.append(row[3])
                age.append(row[4])
                patient_status.append(row[6])
                residence.append(row[7])
                confirm_status.append(row[8])

    with open('building_list_eng.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)

        # # 以迴圈輸出每一列
        for row in rows:
            if first_row:
                first_row = False
                continue
            if not first_row:
                district.append(row[0])
                building_name.append(row[1])
                case_number_1.append(row[3])




class extract_info:
    def __init__(self, object_name, case_number=None):
        self.headers = 'case_number, report_date, date_of_onset, gender, age, patient_status, residence, confirm_status, date, confirmed_case_per_day, death_cases, discharge_cases, probable_cases, critical_condition_cases'
        self.object = object_name
        self.case = case_number
        self.result = list()

        switcher = {
            'case_number':covid_dataset.case_number,
            'report_date':covid_dataset.report_date,
            'date_of_onset':covid_dataset.date_of_onset,
            'gender':covid_dataset.gender,
            'age':covid_dataset.age,
            'patient_status':covid_dataset.patient_status,
            'residence':covid_dataset.residence,
            'confirm_status':covid_dataset.confirm_status,
            'date':covid_dataset.date,
            'confirmed_case_per_day':covid_dataset.confirmed_case_per_day,
            'death_cases':covid_dataset.death_cases,
            'discharge_cases':covid_dataset.discharge_cases,
            'probable_cases':covid_dataset.probable_cases,
            'critical_condition_cases':covid_dataset.critical_condition_cases,
        }

        if self.case == None:
            # return switcher.get(self.object)
            self.result=switcher.get(self.object)

        elif self.case >= 0 and isinstance(self.case, int):
            # return switcher.get(self.object)[self.case+1]
            self.result = switcher.get(self.object)[self.case+1]
