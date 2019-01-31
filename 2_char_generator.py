from numpy import arange
from captcha.image import ImageCaptcha
import random as rand
import sys


def generate_dataset(samples):
    
    combinations=[str(i)+str(j) for i in range(10) for j in range(10)]
   # print(len(combinations))
    counters=[0 for i in combinations]
   #print(len(counters))
    
    for i in range(samples):
        index=rand.randint(0,len(combinations)-1)
        #print(index)
        
        image = ImageCaptcha(fonts=['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'])
        image.write(combinations[index],combinations[index]+'-'+str(counters[index])+'.png')
        counters[index]=counters[index]+1

def main():

    if len(sys.argv) == 2:
        samples = int(sys.argv[1])
        print("generating " + sys.argv[1] + " captcha per char...")
        generate_dataset(samples)
        print("done")
    else:
        print("Correct use: python3 random_captcha_generator.py [captcha-per-char]")

if __name__ == "__main__":
    main()