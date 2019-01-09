from numpy import arange
from captcha.image import ImageCaptcha
from string import ascii_lowercase,ascii_uppercase
import random as rand


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
    generate_dataset(10)
    
   
if __name__ == "__main__":
    main()