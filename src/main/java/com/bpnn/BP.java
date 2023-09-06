package com.bpnn;

/**
 * BP神经网络类
 * 使用了附加动量法进行优化
 * 主要使用方法：
 *     初始化：   BP bp = new BP(new int[]{int,int*n,int})  //第一个int表示输入层，中间n个int表示隐藏层，最后一个int表示输出层
 *     训练： bp.train(double[],double[])               //第一个double[]表示输入，第二个double[]表示期望输出
 *     测试       int result = bp.test(double[])            //参数表示输入,返回值表示输出层最大权值
 *     另有设置学习率和动量参数方法
 */
import java.util.Random;

public class BP {

    private final double[][] layers;//输入层、隐含层、输出层
    private final double[][] deltas;//每层误差
    private final double[][][] weights;//权值
    private final double[][][] prevUptWeights;//更新之前的权值信息
    private final double[] target;   //预测的输出内容

    private double eta;        //学习率
    private double momentum;    //动量参数

    private final Random random;  //主要是对权值采取的是随机产生的方法

    //初始化
    public BP(int[] size, double eta, double momentum) {
        int len = size.length;
        //初始化每层
        layers = new double[len][];
        for(int i = 0; i<len; i++) {
            layers[i] = new double[size[i] + 1];
        }
        //初始化预测输出
        target = new double[size[len - 1] + 1];

        //初始化隐藏层和输出层的误差
        deltas = new double[len - 1][];
        for(int i = 0; i < (len - 1); i++) {
            deltas[i] = new double[size[i + 1] + 1];
        }

        //使每次产生的随机数都是第一次的分配，这是有参数和没参数的区别
        random = new Random(100000);
        //初始化权值
        weights = new double[len - 1][][];
        for(int i = 0; i < (len - 1); i++) {
            weights[i] = new double[size[i] + 1][size[i + 1] + 1];
        }
        randomizeWeights(weights);

        //初始化更新前的权值
        prevUptWeights = new double[len - 1][][];
        for(int i = 0; i < (len - 1); i++) {
            prevUptWeights[i] = new double[size[i] + 1][size[i + 1] + 1];
        }

        this.eta = eta;             //学习率
        this.momentum = momentum;   //动态量
    }

    //随机产生神经元之间的权值信息
    private void randomizeWeights(double[][][] matrix) {
        for (int i = 0, len = matrix.length; i != len; i++) {
            for (int j = 0, len2 = matrix[i].length; j != len2; j++) {
                for(int k = 0, len3 = matrix[i][j].length; k != len3; k++) {
                    double real = random.nextDouble();    //随机分配着产生0-1之间的值
                    matrix[i][j][k] = random.nextDouble() > 0.5 ? real : -real;
                }
            }
        }
    }

    //初始化输入层，隐含层，和输出层
    public BP(int[] size) {
        this(size, 0.25, 0.9);
    }

    //训练数据
    public void train(double[] trainData, double[] target) {
        loadValue(trainData,layers[0]);       //加载输入的数据
        loadValue(target,this.target);         //加载输出的结果数据
        forward();                  //向前计算神经元权值(先算输入到隐含层的，然后再算隐含到输出层的权值)
        calculateDelta();           //计算误差逆传播值
        adjustWeight();             //调整更新神经元的权值
    }

    //加载数据
    private void loadValue(double[] value,double [] layer) {
        if (value.length != layer.length - 1)
            throw new IllegalArgumentException("Size Do Not Match.");
        System.arraycopy(value, 0, layer, 1, value.length);  //调用系统复制数组的方法(存放输入的训练数据)
    }

    //向前计算(先算输入到隐含层的，然后再算隐含到输出层的权值)
    private void forward() {
        //计算隐含层到输出层的权值
        for(int i = 0; i < (layers.length - 1); i++) {
            forward(layers[i], layers[i+1], weights[i]);
        }
    }

    //计算每一层的误差(因为在BP中，要达到使误差最小)(就是逆传播算法，书上有P101)
    private void calculateDelta() {
        outputErr(deltas[deltas.length-1],layers[layers.length - 1],target);   //计算输出层的误差(因为要反过来算，所以先算输出层的)

        for(int i = (layers.length - 1); i > 1; i--) {
            hiddenErr(deltas[i - 2/*输入层没有误差*/],layers[i - 1],deltas[i - 1],weights[i - 1]);   //计算隐含层的误差
        }
    }

    //更新每层中的神经元的权值信息
    private void adjustWeight() {
        for(int i = (layers.length - 1); i > 0; i--) {
            adjustWeight(deltas[i - 1], layers[i - 1], weights[i - 1], prevUptWeights[i - 1]);
        }
    }

    //向前计算各个神经元的权值(layer0：某层的数据,layer1：下一层的内容，weight：某层到下一层的神经元的权值)
    private void forward(double[] layer0, double[] layer1, double[][] weight) {
        layer0[0] = 1.0;//给偏置神经元赋值为1（实际上添加了layer1层每个神经元的阙值）简直漂亮!!!
        for (int j = 1, len = layer1.length; j != len; ++j) {
            double sum = 0;//保存权值
            for (int i = 0, len2 = layer0.length; i != len2; ++i) {
                sum += weight[i][j] * layer0[i];
            }
            layer1[j] = sigmoid(sum);  //调用神经元的激活函数来得到结果(结果肯定是在0-1之间的)
        }
    }

    //计算输出层的误差(delte:误差，output:输出，target：预测输出)
    private void outputErr(double[] delte, double[] output,double[] target) {
        for (int idx = 1, len = delte.length; idx != len; ++idx) {
            double o = output[idx];
            delte[idx] = o * (1d - o) * (target[idx] - o);
        }
    }

    //计算隐含层的误差(delta:本层误差,layer：本层,delta1：下一层误差,weights：权值)
    private void hiddenErr(double[] delta, double[] layer, double[] delta1, double[][] weights) {
        for (int j = 1, len = delta.length; j != len; ++j) {
            double o = layer[j];  //神经元权值
            double sum = 0;
            for (int k = 1, len2 = delta1.length; k != len2; ++k)  //由输出层来反向计算
                sum += weights[j][k] * delta1[k];
            delta[j] = o * (1d - o) * sum;
        }
    }

    //更新每层中的神经元的权值信息(这也就是不断的训练过程)
    private void adjustWeight(double[] delta, double[] layer, double[][] weight, double[][] prevWeight) {
        layer[0] = 1;
        for (int i = 1, len = delta.length; i != len; ++i) {
            for (int j = 0, len2 = layer.length; j != len2; ++j) {
                //通过公式计算误差限=(动态量*之前的该神经元的阈值+学习率*误差*对应神经元的阈值)，来进行更新权值
                double newVal = momentum * prevWeight[j][i] + eta * delta[i] * layer[j];
                weight[j][i] += newVal;  //得到新的神经元之间的权值
                prevWeight[j][i] = newVal;  //保存这一次得到的权值，方便下一次进行更新
            }
        }
    }

    //我这里用的是sigmoid激活函数，当然也可以用阶跃函数，看自己选择吧
    private double sigmoid(double val) {
        return 1d / (1d + Math.exp(-val));
    }

    //测试神经网络
    public int test(double[] inData) {
        if (inData.length != layers[0].length - 1)
            throw new IllegalArgumentException("Size Do Not Match.");
        System.arraycopy(inData, 0, layers[0], 1, inData.length);
        forward();
        return getNetworkOutput();
    }

    //返回最后的输出层的结果
    private int getNetworkOutput() {
        int len = layers[layers.length - 1].length;
        double[] temp = new double[len - 1];
        for (int i = 1; i != len; i++)
            temp[i - 1] = layers[layers.length - 1][i];
        //获得最大权值下标
        double max = temp[0];
        int idx = -1;
        for (int i = 0; i <temp.length; i++) {
            if (temp[i] >= max) {
                max = temp[i];
                idx = i;
            }
        }
        return idx;
    }

    //设置学习率
    public void setEta(double eta) {
        this.eta = eta;
    }

    //设置动量参数
    public void setMomentum(double momentum){
        this.momentum = momentum;
    }
}
