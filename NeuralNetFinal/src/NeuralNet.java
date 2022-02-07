import java.util.Arrays;

public class NeuralNet {
    /*LAYER DETAILS*/

    /*
    NETWORK_LAYER_SIZES - Number of neurons in each layer
    INPUT_SIZE - Size of input
    OUTPUT_SIZE - Size of output
    NETWORK_SIZE - Total number of layers in the network
     */

    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    /*DATA DETAILS OF EACH LAYER*/

    /*
    bias[layer][neuron]
    output[layer][neuron]
    weights[layer][neuron][previous neuron]
    error_signal[layer][neuron]
    output_derivative[layer][neuron] - binary
    output_derivative_bipolar[layer][neuron] - bipolar
    */

    private double[][] bias;
    private double[][] output;
    private double[][][] weights;

    private double[][] error;
    private double[][] error_signal;
    private double[][] output_derivative;
    private double[][] output_derivative_bipolar;

    /*CONSTRUCTOR*/
    public NeuralNet(int... NETWORK_LAYER_SIZES) {

        /*LAYER DETAILS*/
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.error = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];
        this.output_derivative_bipolar = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative_bipolar[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], 1, 1);// Initial value of bias is assigned as 1
            // System.out.println("bias is: "+ Arrays.toString(bias[i]));

            if (i > 0) {
                //assigned random value between given ranges to weights
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -0.5, 0.5);
            }
        }
    }

    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) return null;

        /*Input neuron*/
        this.output[0] = input;

        /*Hidden and output neuron*/
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    //for previous neuron, layer is layer - 1
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }

                //bipolar
                output[layer][neuron] = bipolarsigmoid(sum);
                output_derivative_bipolar[layer][neuron] = (0.5 * (1 + output[layer][neuron]) * (1 - output[layer][neuron]));
            }
        }
        return output[NETWORK_SIZE - 1];
    }

    public double bipolarsigmoid(double x) {
        return ((2d / (1 + Math.exp(-x))) - 1);
    }

    public double[] backpropError(double[] target) {

        /*Output layer neurons*/
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            //error of neurons in the output layer
            for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 2]; prevNeuron++) {

                //bipolar
                error[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) * (output[NETWORK_SIZE - 1][neuron] - target[neuron]);
                error_signal[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) * output_derivative_bipolar[NETWORK_SIZE - 1][neuron];
                //System.out.println("Error: "+error_signal[NETWORK_SIZE-1][neuron]);
            }
        }

        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    double sum = 0;
                    for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                        sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                    }
                    //bipolar
                    this.error_signal[layer][neuron] = sum * output_derivative_bipolar[layer][neuron]; //error of neurons in the hidden layer
                }
            }
        }
        return error[NETWORK_SIZE - 1];
    }

    public void updateWeights(double learning_rate, double momentum) {

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    double delta_weight = -learning_rate * output[layer - 1][prevNeuron] * error_signal[layer][neuron];
                    weights[layer][neuron][prevNeuron] += delta_weight + delta_weight * momentum;
                }
                double delta_bias = -learning_rate * error_signal[layer][neuron];
                bias[layer][neuron] += delta_bias;
            }
        }
    }

    public void train(double[] input, double[] target, double learning_rate, double momentum) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE)
            return;
        calculate(input); //calculate the output
        backpropError(target);
        updateWeights(learning_rate, momentum);
    }


    public static void main(String[] args) {
        NeuralNet net = new NeuralNet(2, 4, 1);

        /*Bipolar input*/
        double[] input_1 = new double[]{-1, -1};
        double[] target_1 = new double[]{-1};
        double[] input_2 = new double[]{-1, 1};
        double[] target_2 = new double[]{1};
        double[] input_3 = new double[]{1, -1};
        double[] target_3 = new double[]{1};
        double[] input_4 = new double[]{1, 1};
        double[] target_4 = new double[]{-1};

        double[] error_value_1;
        double[] error_value_2;
        double[] error_value_3;
        double[] error_value_4;
        double[] target_error = new double[]{0.05};
        double[] error = new double[]{0};

        System.out.println("Epoch,Error");
        for (int i = 0; i < 10000; i++) {
            int epoch = i + 1;
            if (i > 0) {
                if (error[0] >= target_error[0]) {
                    net.train(input_1, target_1, 0.2, 0.9);
                    error_value_1 = net.backpropError(target_1);
                    net.train(input_2, target_2, 0.2, 0.9);
                    error_value_2 = net.backpropError(target_2);
                    net.train(input_3, target_3, 0.2, 0.9);
                    error_value_3 = net.backpropError(target_3);
                    net.train(input_4, target_4, 0.2, 0.9);
                    error_value_4 = net.backpropError(target_4);

                    //Completion of one epoch
                    error[0] = (error_value_1[0] + error_value_2[0] + error_value_3[0] + error_value_4[0]) * 0.5;
                    System.out.println(epoch + "," + error[0]);
                } else {
                    break;
                }
            } else {
                net.train(input_1, target_1, 0.2, 0.9);
                error_value_1 = net.backpropError(target_1);
                net.train(input_2, target_2, 0.2, 0.9);
                error_value_2 = net.backpropError(target_2);
                net.train(input_3, target_3, 0.2, 0.9);
                error_value_3 = net.backpropError(target_3);
                net.train(input_4, target_4, 0.2, 0.9);
                error_value_4 = net.backpropError(target_4);

                //Completion of one epoch
                error[0] = (error_value_1[0] + error_value_2[0] + error_value_3[0] + error_value_4[0]) * 0.5;
                System.out.println(epoch + "," + error[0]);
            }
        }
        double[] o = net.calculate(input_1);
        System.out.println("Output: " + Arrays.toString(o));
    }

    public void train(double[] input, double[] target, double v) {
    }
}


