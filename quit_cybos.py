"""
    Cybos Plus를 종료시키는 스크립트

    Created on 2021.09.04
    @author: Younghyun Kim
"""
import os


if __name__ == "__main__":
    os.system('taskkill /IM ncStarter* /F /T')
    os.system('taskkill /IM CpStart* /F /T')
    os.system('taskkill /IM DibServer* /F /T')
    os.system('wmic process where ' +
              '"name like \'%ncStarter%\'" call terminate')
    os.system('wmic process where ' +
              '"name like \'%CpStart%\'" call terminate')
    os.system('wmic process where ' +
              '"name like \'%DibServer%\'" call terminate')
