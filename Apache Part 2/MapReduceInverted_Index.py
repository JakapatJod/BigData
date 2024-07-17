from mrjob.job import MRJob
from mrjob.step import MRStep

class MapReduceFilter(MRJob):
    def mapper(self, _, line):
        if 'status_id' not in line:
            data = line.split(',')
            datetime = data[2].strip()
            year = datetime.split(' ')[0].split('/')[2]
            date = datetime.split(' ')[0].split('/')[0]
            status_type = data[1].strip()
            num_reactions = data[3].strip()
            
            yield status_type,num_reactions
            # if year == '2018':
            #     if status_type =='video':
            #         yield (year,'video'), data
            #     elif status_type =='photo':
            #         yield (year,'photo'), data
            #     elif status_type =='link':
            #         yield (year,'link'), data
            #     elif status_type =='status':
            #         yield (year,'status'), data
    def reducer(self, key, values):
        lval = []
        for react in values:
            lval.append(react)
        yield key, lval
                    
if (__name__ == '__main__'):
    MapReduceFilter.run()