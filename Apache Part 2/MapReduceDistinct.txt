from mrjob.job import MRJob

class MapReduceDistinct(MRJob):

    def mapper(self, _, line):


        if 'status_id' not in line:

            data = line.split(",")
            datetime = data[2].strip()
            date = datetime.split(' ')[0].split('/')[0]
            year = datetime.split(' ')[0].split('/') [2]

            if year == '2018':
                yield date, None

    def reducer(self, key, values):

        yield key, None

if __name__ == '__main__':
    MapReduceDistinct.run()