import com.bpnn.BP;
import com.bpnn.BPNN;
import com.bpnn.ImageUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Test {
    public static void main(String[] args) {
        BP bp = new BP(new int[]{28 * 28, 16, 16, 10},0.025,0.67);

        //训练
        xl2(bp,
                new String[]{"E:\\Image训练集\\训练\\img-0", "E:\\Image训练集\\训练\\img-1", "E:\\Image训练集\\训练\\img-2", "E:\\Image训练集\\训练\\img-3", "E:\\Image训练集\\训练\\img-4", "E:\\Image训练集\\训练\\img-5", "E:\\Image训练集\\训练\\img-6", "E:\\Image训练集\\训练\\img-7", "E:\\Image训练集\\训练\\img-8", "E:\\Image训练集\\训练\\img-9"},
                new double[][][]{
                        {{0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
                        {{0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
                        {{0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
                        {{0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001}},
                        {{0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001}},
                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001, 0.001}},
                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001, 0.001}},
                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001, 0.001}},
                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999, 0.001}},
                        {{0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.999}}
                });

        double v = testMeList(bp, "E:\\Image训练集\\测试");
        System.out.println(v);

    }


    public static void xl2(BP bpnn, String filePath[], double value[][][]) {
        ArrayList<File[]> arr = new ArrayList<>();

        for (int i = 0; i < filePath.length; ++i) {
            String name = filePath[i].split("-")[1];
            //URL resource = Main.class.getClassLoader().getResource(filePath[i]);
            File file = new File(filePath[i]);
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
                bpnn.train(input[0],value[i][0]);
            }
            if (cc)
                break;
            ++sw;
        }

    }


    /**
     * 自带数据得测试
     * @param bpnn
     * @param filePath
     * @return 正确率
     */
    public static double testMeList(BP bpnn,String filePath){
        File file = new File(filePath);
        File[] files1 = file.listFiles(); //测试数字集文件夹
        int value[] = new int[files1.length]; //期望值
        ArrayList<List<File>> imageList  = new ArrayList<>(); //用于存储测试集

        for (int i = 0; i < files1.length; i++) {
            //赋值期望
            int num = Integer.parseInt(files1[i].getName().split("-")[1]);
            value[i] =num;
            File[] files2 = files1[i].listFiles();
            List<File> files = Arrays.stream(files2).toList();
            imageList.add(files);
        }

        int noSum =0;
        int yesSum =0;
        for (int x = 0; x < 100; x++) {
            //开始测试
            for (int i = 0; i < imageList.size(); i++) {
                int index = (int) Math.random() * imageList.get(i).size();
                int test = bpnn.test(ImageUtil.readImageBinary(imageList.get(i).get(index).getPath())[0]);
                if (i == test) {
                    System.out.println("Yes");
                    ++yesSum;
                } else {
                    System.out.println("No");
                    ++noSum;
                }

            }
        }
        double v = (double) yesSum / (noSum + yesSum);
        return v;
    }
}
