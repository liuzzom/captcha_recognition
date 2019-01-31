from numpy import arange
from captcha.image import ImageCaptcha
import random as rand
import sys


def generate_dataset(samples):

    for i in arange(0, 10):
        for j in arange(0,samples):
            image = ImageCaptcha(fonts=['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'])
            image.write(str(i), str(i)+'-'+str(j)+'.png')

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
