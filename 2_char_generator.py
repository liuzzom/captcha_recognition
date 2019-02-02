from numpy import arange
from captcha.image import ImageCaptcha
import random as rand
import sys


def generate_dataset(samples):
    
    combinations=[str(i)+str(j) for i in range(10) for j in range(10)]
    counters=[0 for i in combinations]
    for i in range(samples):
        index=rand.randint(0,len(combinations)-1)
        
        image = ImageCaptcha(fonts=['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'])
        image.write(combinations[index],combinations[index]+'-'+str(counters[index])+'.png')
        counters[index]=counters[index]+1
        
        
    def generate_dataset_with_0(samples):
        combinations=[str(i)+'0' for i in range(10)]
        counters=[0 for i in combinations]
        for i in range(samples):
            index=rand.randint(0,len(combinations)-1)
            image = ImageCaptcha(fonts=['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'])
            image.write(combinations[index],combinations[index]+'-'+str(counters[index])+'.png')
            counters[index]=counters[index]+1  

def main():

    if len(sys.argv) == 2:
        samples = int(sys.argv[1])
        print("generating " + sys.argv[1] + " captchas...")
        generate_dataset(samples)
        print("done")
    else:
        print("Correct use: python3 random_captcha_generator.py [captcha-num]")

if __name__ == "__main__":
    main()