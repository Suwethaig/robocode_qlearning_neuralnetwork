public class NetworkTools {

    /* Returns a 1D array with every value as initial value */
    public static double[] createArray(int size, double init_value) {
        if (size < 1) {
            return null;
        }
        double[] ar = new double[size];
        for (int i = 0; i < size; i++) {
            ar[i] = init_value;
        }
        return ar;
    }

    /* Returns a 1D array and each value of the array is randomly chosen between the lower and the upper bound */
    public static double[] createRandomArray(int size, double lower_bound, double upper_bound) {
        if (size < 1) {
            return null;
        }
        double[] ar = new double[size];
        for (int i = 0; i < size; i++) {
            ar[i] = randomValue(lower_bound, upper_bound);
        }
        return ar;
    }

    /* Returns a 2D array and each value of the array is randomly chosen between the lower and the upper bound */
    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound) {
        if (sizeX < 1 || sizeY < 1) {
            return null;
        }
        double[][] ar = new double[sizeX][sizeY];
        for (int i = 0; i < sizeX; i++) {
            ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }
        return ar;
    }

    /* Returns a value between the lower and upper bound */
    public static double randomValue(double lower_bound, double upper_bound) {
        return Math.random() * (upper_bound - lower_bound) + lower_bound;
    }

    /* Returns a integer array and each value of the array can never be repeated */
    public static Integer[] randomValues(int lowerBound, int upperBound, int amount) {
        lowerBound--;
        if (amount > (upperBound - lowerBound)) {
            return null;
        }
        Integer[] values = new Integer[amount];
        for (int i = 0; i < amount; i++) {
            int n = (int) (Math.random() * (upperBound - lowerBound + 1) + lowerBound);
            while (containsValue(values, n)) {
                n = (int) (Math.random() * (upperBound - lowerBound + 1) + lowerBound);
            }
            values[i] = n;
        }
        return values;
    }

    /* For any array of any data type it checks if a data is contained in the array */
    public static <T extends Comparable<T>> boolean containsValue(T[] ar, T value) {
        for (int i = 0; i < ar.length; i++) {
            if (ar[i] != null) {
                if (value.compareTo(ar[i]) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    /* Finds the index of highest value elements of an array */
    public static int indexOfHighestValue(double[] values) {
        int index = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[index]) {
                index = i;
            }
        }
        return index;
    }
}