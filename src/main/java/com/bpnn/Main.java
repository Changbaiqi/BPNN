package com.bpnn;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {

        BPNN bpnn = BPNN.init(28 * 28, new int[]{16, 16}, 10, 0.025);

        //读取并设置权值
        bpnn.setWeights(BPNN.readWeights("E:/IntelliJ_WorkSpace/BPNNImage", "weightsTest53.txt"));
//        int test = test(bpnn, ImageUtil.readImageBinary("E:/IntelliJ_WorkSpace/BPNNImage/src/main/resources/5-6-test.png"));
//        System.out.println(test);
//        bpnn.initRandomWeight();
//        //训练
//        xl2(bpnn,
//                new String[]{"img-0", "img-1", "img-2", "img-3", "img-4", "img-5", "img-6", "img-7", "img-8", "img-9"},
//                new double[][][]{
//                        {{0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001}},
//                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999}}
//                });
        float v = testList(bpnn, "E:/IntelliJ_WorkSpace/BPNNImage/src/main/resources");
    }

    public static float testList(BPNN bpnn, String filePath) {
        File file = new File(filePath);
        List<File> files = Arrays.stream(file.listFiles()).filter(file1 -> file1.getName().endsWith(".png")).toList();
        int yes = 0;
        int no = 0;
        for (int i = 0; i < files.size(); i++) {
            System.out.println("文件名："+files.get(i).getName());
            double[][] doubles = ImageUtil.readImageBinary(files.get(i).getPath());
            System.out.println("输入：" + Arrays.deepToString(doubles));
            int test = test(bpnn, doubles);
            String[] split = files.get(i).getName().split("-");
            //bpnn.printNeural();
            System.out.println("实际输出：" + test + "----------------" + "期望值为：" + split[0]);
            if (Integer.parseInt(split[0]) == test)
                ++yes;
            else
                ++no;
            System.out.println("\n\n");
            //System.out.println(files.get(i).getName());
        }
        System.out.println("总个数：" + (yes + no));
        System.out.println("正确个数：" + yes);
        System.out.println("错误个数：" + no);
        float v = (float) yes / (yes + no);
        System.out.println("正确率：" + v);
        return v;

    }

    public static void xl(BPNN bpnn, double out[][], String filePath) {
        URL resource = Main.class.getClassLoader().getResource(filePath);
        File file = new File(resource.getPath());
        File[] files = file.listFiles();
        for (int i = 0; i < files.length; i++) {
            //读取测试集图片
            double[][] input = ImageUtil.readImageBinary(files[i].getPath());
            //输入
            bpnn.input(input);
            //向前传播
            bpnn.formDiffuse();
            //向后传播
            bpnn.backPropagation(out);
        }
    }

    public static void xl2(BPNN bpnn, String filePath[], double value[][][]) {
        ArrayList<File[]> arr = new ArrayList<>();

        for (int i = 0; i < filePath.length; ++i) {
            String name = filePath[i].split("-")[1];
            URL resource = Main.class.getClassLoader().getResource(filePath[i]);
            File file = new File(resource.getPath());
            File[] files = file.listFiles();
            arr.add(files);
        }

        int sw = 1;
        while (sw >= 0) {
            System.out.println("第"+sw+"轮训练");
            boolean cc = true;
            for (int i = 0; i < arr.size(); i++) {
                if (arr.get(i).length - 1 >= sw)
                    cc = false;
                else
                    continue;
                if (sw == 4100) {
                    cc = true;
                    break;
                }
                //读取测试集图片
                double[][] input = ImageUtil.readImageBinary(arr.get(i)[sw].getPath());
                //输入
                bpnn.input(input);
                //向前传播
                bpnn.formDiffuse();
                //向后传播
                bpnn.backPropagation(value[i]);
            }
            if (cc)
                break;
            ++sw;
        }

    }

    public static int test(BPNN bpnn, double input[][]) {
        //输入
        bpnn.input(input);
        //向前传播
        bpnn.formDiffuse();
        //输出输出层
        //bpnn.printOut();
        double[][] doubles = bpnn.getNeural().get(bpnn.getNeural().size() - 1);
        int maxIndex = 0;
        double maxNum = 0;
        for (int i = 0; i < doubles[0].length; i++) {
            if (doubles[0][i] > maxNum) {
                maxNum = doubles[0][i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


}