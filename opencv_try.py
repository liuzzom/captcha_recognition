import cv2
import sys


def main():
    path = sys.argv[1]
    #filename=sys.argv[2]
    im = cv2.imread(path, 0)
    _,img=cv2.threshold(im,210,255,cv2.THRESH_BINARY)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    


if __name__ == "__main__":
    main()