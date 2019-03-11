# -*- coding:utf-8 -*-


__author__ = 'XH-001'

import unicodecsv
import datetime

# 查看CSV数据
"""
enrollments = []
f = open('enrollments.csv', 'rb')
reader = unicodecsv.DictReader(f)

for row in reader:
    enrollments.append(row)
"""

#导入CSV数据


def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)


enrollments = read_csv('enrollments.csv')
daily_engagement = read_csv('daily-engagement.csv')
project_submissions = read_csv('project-submissions.csv')

#print(len(enrollments))
#print(len(daily_engagement))
#print(len(project_submissions))

# 识别唯一学生人数,没有使用函数
"""
engagement_num_unique_students = set()
for enrollment in enrollments:
        engagement_num_unique_students.add(enrollment['account_key'])
len(engagement_num_unique_students)
"""

# 采用定义的方式定义识别唯一学生的函数


def get_unique_students(data, keyname):
    unique_students = set()
    for data_point in data:
        unique_students.add(data_point[keyname])
    return unique_students


engagement_num_unique_students = get_unique_students(enrollments, 'account_key')
unique_engagement_students = get_unique_students(daily_engagement, 'account_key')
unique_project_submitters = get_unique_students(project_submissions, 'account_key')
#len(unique_project_submitters)
#len(unique_engagement_students)
#len(engagement_num_unique_students)

#查出Udacity的测试账号
udacity_test_accounts = set()
for enrollment in enrollments:
    if enrollment['is_udacity'] == "True":
        udacity_test_accounts.add(enrollment['account_key'])
#len(udacity_test_accounts)

#移除Udacity中的测试账号


def remove_udacity_accounts(data):
    non_udacity_data = []
    for data_point in data:
        if data_point['account_key'] not in udacity_test_accounts:
            non_udacity_data.append(data_point)

    return non_udacity_data


non_udacity_enrollments = remove_udacity_accounts(enrollments)
non_udacity_engagement = remove_udacity_accounts(daily_engagement)
non_udacity_submissions = remove_udacity_accounts(project_submissions)

# print(len(non_udacity_enrollments))
# print(len(non_udacity_engagement))
# print(len(non_udacity_submission))

#查看有效非Udacity的注册学生

# paid_students = {}
# for enrollment in non_udacity_enrollments:
#     if (not enrollment['is_canceled'] or enrollment['days_to_cancel'] > 7):
#         account_key = enrollment['account_key']
#         enrollment_date = enrollment['join_date']
#         if (account_key not in paid_students or enrollment_date > paid_students[account_key]):
#             paid_students[account_key] = enrollment_date

# print(len(paid_students))



#students engagement within one week (join date and engagement record


# def within_one_week(join_date, engagement_date):
#     time_delta = engagement_date - join_date
#     return time_delta.days < 7
#
#
# def remove_free_trial_cancels(data):
#     new_data = []
#     for data_point in data:
#         if data_point['account_key'] in paid_students:
#             new_data.append(data_point)
#     return new_data
#
#
# paid_enrollments = remove_free_trial_cancels(non_udacity_enrollments)
# paid_engagement = remove_free_trial_cancels(non_udacity_engagement)
# paid_submissions = remove_free_trial_cancels(non_udacity_submission)
#
# # print(len(paid_enrollments))
# # print(len(paid_engagement))
# # print(len(paid_submissions))
#
# paid_engagement_in_first_week = []
#
# for engagement_record in paid_engagement:
#     account_key = engagement_record['account_key']
#     join_date = paid_students[account_key]
#     engagement_record_date = engagement_record['utc_date']
#
#     if within_one_week(join_date, engagement_record_date):
#         paid_engagement_in_first_week.append(engagement_record)
#
# print(len(paid_engagement_in_first_week))


def within_one_week(join_date, engagement_date):


    engagement_date_int = datetime.datetime.strptime(engagement_date, '%Y-%m-%d')

    join_date_int = datetime.datetime.strptime(join_date, '%Y-%m-%d')
    time_delta = engagement_date_int - join_date_int
    return time_delta.days < 7

# def remove_free_trial_cancels(data):
#     new_data = []
#     for data_point in data:
#         if data_point['account_key'] in paid_students:
#             new_data.append(data_point)
#     return new_data

# paid_enrollments = remove_free_trial_cancels(non_udacity_enrollments)
# paid_engagement = remove_free_trial_cancels(non_udacity_engagement)
# paid_submissions = remove_free_trial_cancels(non_udacity_submissions)

# print len(paid_enrollments)
# print len(paid_engagement)
# print len(paid_submissions)

# paid_engagement_in_first_week = []
# for engagement_record in paid_engagement:
#     account_key = engagement_record['account_key']
#     join_date = paid_students[account_key]
#     engagement_record_date = engagement_record['utc_date']
#
#     if within_one_week(join_date, engagement_record_date):
#         paid_engagement_in_first_week.append(engagement_record)
#
# print(len(paid_engagement_in_first_week))

within_one_week('2019/01/25','2019/01/23')