package com.bpnn;

import java.io.*;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BPNN {
    private double studyRate; //学习率
    private ArrayList<double[][]> resNeural; //临时存储的网络
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

        bpnn.resNeural = new ArrayList<>();
        bpnn.neural = new ArrayList<>();
        bpnn.neural.add( new double[1][inputNeuralSum]); //初始化输入层
        bpnn.resNeural.add(new double[1][inputNeuralSum]);

        //初始化隐含层
        for(int i=0 ; i < implyNeuralSum.length ; ++i){
            int i1 = implyNeuralSum[i];
            bpnn.resNeural.add(new double[1][i1]);
            bpnn.neural.add(new double[1][i1]);
        }

        //初始化输出层
        bpnn.resNeural.add(new double[1][outNeuralSum]);
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
        double v = 2;
        while (v>=1 || v<=-1){
            v = new Random().nextGaussian() * (1d / (Math.sqrt(num)));
        }
        return v;
    }


    /**
     * 输入层输入
     * @param inputNeural 输入的数据
     */
    public void input(double inputNeural[][]){
        //输入层值：
        for(int i=0 ; i < inputNeural[0].length ; ++i){
            resNeural.get(0)[0][i] = inputNeural[0][i];
            neural.get(0)[0][i] = inputNeural[0][i];
        }
    }

    /**
     * 向前传播
     */
    public void formDiffuse(){
        //向前传播计算矩阵
        for (int i = 0; i < neural.size() - 1; i++) {
            formProCom(neural.get(i), weights.get(i),neural.get(i+1));
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
            backDiffCom(diffList.get(i),weights.get(i),diffList.get(i+1));
        }

        //这部分逆向传播是从前向后的传播计算
        for(int i =0 ; i< neural.size()-1 ; ++i){
            for(int y = 0; y < neural.get(i)[0].length; ++y){
                double Ll = neural.get(i)[0][y];
//                double Ll = resNeural.get(i)[0][y];
                for(int x =0 ; x < neural.get(i+1)[0].length ;++x){
                    double oldWeight = weights.get(i)[y][x];
                    double Rer = diffList.get(i+1)[0][x];
                    double Rl = neural.get(i+1)[0][x];
//                    double Rl = resNeural.get(i+1)[0][x];
                    double rate = -Rer*Rl*(1-Rl)*Ll;
//                    double rate = -2*(Rer)*(Rl)*(1-Rl);
                    weights.get(i)[y][x] = oldWeight - rate*studyRate;
                    //weights.get(i)[y][x] = oldWeight+1;
                }
            }
        }



    }

    /**
     * 向前传播计算函数
     * @param leftNeural
     * @param weights
     * @return
     */
    private void formProCom(double leftNeural[][],double weights[][],double rightNeural[][]){

        for (int y = 0; y < rightNeural[0].length; y++) {
            double res =0;
            for (int x = 0; x < leftNeural[0].length; x++) {
                res+= leftNeural[0][x]*weights[x][y];
            }
            rightNeural[0][y] = sigmoid(res);
        }
    }

    /**
     * 向后算计差值
     * @param leftDiff
     * @param weights
     * @param rightDiff
     */
    private void backDiffCom(double leftDiff[][],double weights[][],double rightDiff[][]){
        for (int y = 0; y < leftDiff[0].length; y++) {
            double res =0;
            for (int x = 0; x < rightDiff[0].length; x++) {
                res += (rightDiff[0][x]*weights[y][x]);
            }
            leftDiff[0][y] = res;
        }
    }

    /**
     * sigmoid函数
     * @param value
     * @return
     */
    private double sigmoid(double value){
//        return 1d/(1d+ Math.exp(-value));
        if( value >=0)
            return 1/(1+Math.exp(-value));
        else
            return Math.exp(value)/(1+Math.exp(value));
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

    /**
     * 获取输出层
     * @return
     */
    public double[][] getOut(){
        return neural.get(neural.size()-1);
    }
    public double[] getOutMaxIndexAndValue(){
        double index =-1;
        double maxValue=0;
        double[] doubles = getOut()[0];
        for (int i = 0; i < doubles.length; i++) {
            if(doubles[i]>maxValue){
                index = i;
                maxValue = doubles[i];
            }
        }
        return new double[]{index,maxValue};
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
