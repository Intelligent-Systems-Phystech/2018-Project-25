class Parser:
    def __init__(self, dlst):
        self.dirlist = dlst

    def parse(self, rootName): 
        import re
        dirlist = self.dirlist
        ret = {}
        rootName +='/'
        for dirname in dirlist:
            poslistname = rootName + dirname + '/' + 'pos.lst'
            poslst = open(poslistname,'r')
            anfname = rootName + dirname + '/' + 'annotations.lst'
            anflst = open(anfname, 'r')
            for line in poslst:  # for every img with pedestain(s)
                filename = rootName + anflst.readline()[:-1]
                # filename = 'Test/annotations/crop_000001.txt'
                annot = open(filename, 'r')
                xylist = []
                for ln in annot:
                    if ln.startswith('Bounding'):  
                        ln = re.split(r':', ln)
                        ln = re.split(r'[^1234567890]', ln[1])
                        # print ln
                        while '' in ln:
                            ln.remove('')
                        xy_min = (int(ln[0]), int(ln[1]))
                        xy_max = (int(ln[2]), int(ln[3]))
                        objcords = (xy_min, xy_max)
                        xylist.append(objcords)
                ret[line[:-1]] = xylist
        return ret
