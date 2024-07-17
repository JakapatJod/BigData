from mrjob.job import MRJob
from mrjob.step import MRStep

class MapReduceBinning(MRJob):
    def mapper(self, _, line):
        if 'status_id' not in line:
            data = line.split(',')
            datetime = data[2].strip()
            year = datetime.split(' ')[0].split('/')[2]
            date = datetime.split(' ')[0].split('/')[0]
            status_type = data[1].strip()

            if year == '2018':
                if status_type =='video':
                    yield (year,'video'), data
                elif status_type =='photo':
                    yield (year,'photo'), data
                elif status_type =='link':
                    yield (year,'link'), data
                elif status_type =='status':
                    yield (year,'status'), data
if (__name__ == '__main__'):
    MapReduceBinning.run()