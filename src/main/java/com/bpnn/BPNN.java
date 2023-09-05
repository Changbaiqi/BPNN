package com.bpnn;

import org.apache.commons.math3.linear.Array2DRowFieldMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BPNN {
    private double studyRate; //学习率
    private ArrayList<double[][]> neural; //用于存储网络
    private ArrayList<double[][]> diffList; //用于存储差值
    private ArrayList<double[][]> weights; //用于存储权重值

    /**
     * 初始化BPNN
     * @param inputNeuralSum 输入层的神经元个数
     * @param implyNeuralSum 每一层隐含层的神经元的个数
     * @param outNeuralSum 输出层的神经元的个数
     * @param studyRate 学习率
     * @return 返回BPNN对象
     */
    public static BPNN init(int inputNeuralSum,int implyNeuralSum[],int outNeuralSum,double studyRate){
        BPNN bpnn = new BPNN();
        bpnn.studyRate = studyRate;

        bpnn.neural = new ArrayList<>();
        bpnn.neural.add( new double[1][inputNeuralSum]); //初始化输入层

        //初始化隐含层
        for(int i=0 ; i < implyNeuralSum.length ; ++i){
            int i1 = implyNeuralSum[i];
            bpnn.neural.add(new double[1][i1]);
        }

        //初始化输出层
        bpnn.neural.add( new double[1][outNeuralSum]);

        //初始化差值存储
        bpnn.diffList = new ArrayList<>();
        for (int i = 0; i < bpnn.neural.size(); i++) {
            bpnn.diffList.add(new double[1][bpnn.neural.get(i)[0].length]);
        }

        //初始化权值存储
        bpnn.weights = new ArrayList<>();
        for (int i = 0; i < bpnn.neural.size()-1; i++) {
            bpnn.weights.add(new double[bpnn.neural.get(i)[0].length][bpnn.neural.get(i+1)[0].length]);
        }
        return bpnn;
    }


    /**
     * 初始化算计权值
     */
    public void initRandomWeight(){
        for (double[][] weight : this.weights) {
            for(int y =0 ; y < weight.length ; ++y){
                for(int x =0 ; x < weight[y].length ; ++x){
                    weight[y][x] = randomWeightGaussian(weight[y].length);
                    //weight[y][x] = randomWeight(weight.length);
                }
            }
        }
    }

    /**
     * 普通随机
     * @param num
     * @return
     */
    private double randomWeight(int num){
        return (Math.random()*(2/(Math.sqrt(num))))+((-1/(Math.sqrt(num))));
    }

    /**
     * 采用高斯随机法随机
     * @param num
     * @return
     */
    private double randomWeightGaussian(int num){
        return new Random().nextGaussian()*(1/(Math.sqrt(num)));
    }


    /**
     * 输入层输入
     * @param inputNeural 输入的数据
     */
    public void input(double inputNeural[][]){
        //输入层值：
        for(int i=0 ; i < inputNeural[0].length ; ++i){
            neural.get(0)[0][i] = inputNeural[0][i];
        }
    }

    /**
     * 向前传播
     */
    public void formDiffuse(){
        //向前传播
        for(int i = 0 ; i < neural.size()-1 ; ++i){
            Array2DRowRealMatrix implyMatrix1 = new Array2DRowRealMatrix(neural.get(i));
            Array2DRowRealMatrix implyMatrix2 = new Array2DRowRealMatrix(weights.get(i));
            Array2DRowRealMatrix multiply1 = implyMatrix1.multiply(implyMatrix2);
            for(int x=0 ;x < multiply1.getRow(0).length ; ++x){
                double sigmoidValue = sigmoid(multiply1.getEntry(0, x));
                neural.get(i+1)[0][x] = sigmoidValue;
            }
        }

    }

    /**
     * 逆向传播
     * @param expectedNumber
     */
    public void backPropagation(double expectedNumber[][]){
        //差值计算-----------------------------------------------------
        for (int i = 0; i < expectedNumber[0].length; i++) {
            diffList.get(diffList.size()-1)[0][i] = expectedNumber[0][i]-neural.get(neural.size()-1)[0][i];
        }
        //隐含层的差值及输入层的差值
        for(int i=diffList.size()-2  ; i >=0 ; --i){
            Array2DRowRealMatrix diffMatRightMat = new Array2DRowRealMatrix(diffList.get(i+1));
            Array2DRowRealMatrix weightMat = new Array2DRowRealMatrix(weights.get(i));

            RealMatrix multiply = diffMatRightMat.multiply(weightMat.transpose());
            for (int i1 = 0; i1 < multiply.getRow(0).length; i1++) {
                diffList.get(i)[0][i1] = multiply.getEntry(0,i1);
            }
            //System.out.println(Arrays.deepToString(diffList.get(i)));
        }


        //逆向传播--------------------------------------------------------------------------
        //输出层到隐含最后一层
//        for(int i=weights.size()-1 ; i >= 0 ; --i) {
//            for (int y = 0; y < weights.get(i).length; ++y) {
//                for (int x = 0; x < weights.get(i)[y].length; ++x) {
//                    double Rer= diffList.get(i+1)[0][x];
//                    double Rl = neural.get(i+1)[0][x];
//                    double Ll = neural.get(i)[0][y];
//                    double rate = -Rer*Rl*(1-Rl)*Ll;
//                    weights.get(i)[y][x] =weights.get(i)[y][x] - rate*studyRate;
//                }
//            }
//        }

        for(int i =0 ; i< neural.size()-1 ; ++i){
            for(int y = 0; y < neural.get(i)[0].length; ++y){
                double Ll = neural.get(i)[0][y];
                for(int x =0 ; x < neural.get(i+1)[0].length ;++x){
                    double oldWeight = weights.get(i)[y][x];
                    double Rer = diffList.get(i+1)[0][x];
                    double Rl = neural.get(i+1)[0][x];
                    double rate = -Rer*Rl*(1-Rl)*Ll;
                    weights.get(i)[y][x] = oldWeight - rate*studyRate;
                }
            }
        }



    }

    /**
     * sigmoid函数
     * @param value
     * @return
     */
    private double sigmoid(double value){
        return 1/(1+ Math.exp(-value));
    }

    public double getStudyRate() {
        return studyRate;
    }

    public ArrayList<double[][]> getNeural() {
        return neural;
    }

    public ArrayList<double[][]> getDiffList() {
        return diffList;
    }

    public void setWeights(ArrayList<double[][]> weightsList) {
        for (int i = 0; i < weights.size(); i++) {
            for (int y = 0; y < weightsList.get(i).length; y++) {
                for (int x = 0; x < weightsList.get(i)[y].length; x++) {
                    weights.get(i)[y][x] = weightsList.get(i)[y][x];
                }
            }
        }
    }

    public ArrayList<double[][]> getWeights() {
        return weights;
    }

    /**
     * 输出神经网络亮度值
     */
    public void printNeural(){
        for (int i = 0; i < neural.size(); i++) {
            System.out.println(Arrays.deepToString(neural.get(i)));
        }
    }

    /**
     * 输出神经网络权值
     */
    public void printWeights(){
        for (int i = 0; i < weights.size(); i++) {
            System.out.println(Arrays.deepToString(weights.get(i)));
        }
    }

    /**
     * 输出神经网络差值
     */
    public void printDiffs(){
        for (int i = 0; i < diffList.size(); i++) {
            System.out.println(Arrays.deepToString(diffList.get(i)));
        }
    }

    public void printOut(){
        System.out.println(Arrays.toString(neural.get(neural.size()-1)[0]));
    }


    /**
     * 读取权重配置文件
     * @param filePath
     * @return
     */
    public static ArrayList<double[][]> readWeights(String filePath,String fileName){
        ArrayList<double[][]> weightList = new ArrayList<>();

        StringBuffer stringBuffer = new StringBuffer();

        //数据读取
        File file = new File(filePath,fileName);
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            String resStr = null;
            while( (resStr = bufferedReader.readLine())!=null){
                stringBuffer.append(resStr);
            }

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        //数据转换
        String[] split = stringBuffer.toString().split("======");
        for(int i=0; i < split.length ; ++i){
            String[] split1 = split[i].split("###");
            double data[][] = new double[split1.length][split1[0].split(",").length];
            for(int y =0 ;y < split1.length ; ++y){
                String[] split2 = split1[y].split(",");
                for (int x =0 ; x < split2.length ; ++x){
                    data[y][x] = Double.parseDouble(split2[x]);
                }
            }
            weightList.add(data);
        }
        //System.out.println(stringBuffer.toString());
        return weightList;
    }

    /**
     * 存储权重配置文件
     * @param weights
     */
    public static void saveWeights(ArrayList<double[][]> weights,String filePath,String fileName){
        StringBuffer stringBuffer = new StringBuffer();
        //数据装载
        for (int i = 0; i < weights.size(); i++) {
            for(int y =0;y < weights.get(i).length ; ++y){
                for (int x =0 ; x < weights.get(i)[y].length ; ++x){
                    stringBuffer.append(weights.get(i)[y][x]);
                    if(x!=weights.get(i)[y].length-1)
                        stringBuffer.append(",");
                }
                if(y!=weights.get(i).length-1)
                    stringBuffer.append("###");
            }
            if(i!=weights.size()-1)
                stringBuffer.append("======");
        }
        //数据存储
        File file = new File(filePath,fileName);
        try {
            file.createNewFile();
            FileWriter writer = new FileWriter(fileName,false);
            BufferedWriter bufferedWriter = new BufferedWriter(writer);
            bufferedWriter.write(stringBuffer.toString());
            bufferedWriter.flush();
            bufferedWriter.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


}
