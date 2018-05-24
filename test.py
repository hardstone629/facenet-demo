import numpy as np

def main():
    ll = [[0.98,0.02,1,2]]
    ar = np.array(ll)
    best_class_indices = np.argmax(ar, axis=1)
    print(best_class_indices)
    print(len(best_class_indices))
    print(ar[0,best_class_indices[0]])

if __name__ == '__main__':
    main()