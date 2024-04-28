import logging
import os.path


class Log:
    def __init__(self, name, level=logging.INFO):
        if os.path.exists("RunLogs/"):
            pass
        else:
            os.mkdir("RunLogs/")
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.level = level
        self.setLogLevel()
        self.fileHandler()
        self.alllogHandler()
        self.log()

    def setLogLevel(self):
        self.logger.setLevel(self.level)

    def fileHandler(self):
        file_handler = logging.FileHandler(f'RunLogs/{self.name}.log', mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def alllogHandler(self):
        file_handler = logging.FileHandler('RunLogs/Full.log', mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self):
        return self.logger
