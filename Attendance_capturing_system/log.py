# Function for Logging Attendance
from datetime import datetime as dt
from dateutil.tz import gettz


def attendanceLog(name):
    current_date = dt.now(tz=gettz('Asia/Kolkata')).strftime('%d.%B.%Y')  # %m : 1, %B : Jan
    path = 'AttendanceDates/'
    current_date_csv = path + current_date + '.csv'
    with open(current_date_csv, 'a+') as f:
        f.seek(0, 0)
        dataList = f.readlines()  # each line contains name
        names = []
        for line in dataList:
            entry = line.split(',')
            names.append(entry[0])
        if name not in names:
            now = dt.now(tz=gettz('Asia/Kolkata'))  # current time
            format = now.strftime('%H:%M:%S')
            f.writelines(f"{name},{format}\n")


if __name__ == '__main__':
    attendanceLog('vivek')
