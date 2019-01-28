from numpy import arange
from captcha.image import ImageCaptcha
from string import ascii_lowercase,ascii_uppercase
import random as rand
import sys


def generate_dataset(samples):
    lowercase=[letter for letter in ascii_lowercase]
    uppercase=[letter for letter in ascii_uppercase]
    numbers=[str(num) for num in arange(0,9)]
    
    symbols=lowercase+uppercase+numbers
     
    for i in arange(0,samples):
        string="".join(rand.choices(symbols,k=4))
        image = ImageCaptcha(fonts=['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'])
        
        image.write(string,string+'.png')

def main():
	
	if len(sys.argv) == 2:
		samples = int(sys.argv[1])
		print("generating " + sys.argv[1] + " images...")
		generate_dataset(samples)
		print("done")
	else:
		print("Correct use: python3 random_captcha_generator.py [num]")
   
if __name__ == "__main__":
    main()