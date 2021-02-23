import os


def generate(split, GT_folder):
    file_ptr = open(split, 'r')
    content_all = file_ptr.read().split('\n')[1:]  
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    file = open('testing_segments.txt','a')
    lines = ''
    line = []
    for (idx, content) in enumerate(content_all):
        file_ptr = open(GT_folder + content, 'r')
        curr_gt = file_ptr.read().split('\n')[:-1]

        initial = curr_gt[0]
        
        for (num,item) in enumerate(curr_gt):
            if item == initial:
                continue
            print(num)
            line.append(num)
            initial = item

        for i in line:
            lines = lines + ' ' + str(i)
        
        file.write(lines)
        file.write('\n')
        line = []
        lines = ''

    file.close()
        
    
if __name__ == "__main__":
    COMP_PATH = './Breakfast-Data/'
    train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split = os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder = os.path.join(COMP_PATH, 'groundTruth/')

    # generate(train_split, GT_folder)
    generate(test_split, GT_folder)


