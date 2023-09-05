import com.bpnn.BPNN;

import java.util.ArrayList;
import java.util.Arrays;

public class Test {
    public static void main(String[] args) {
        BPNN bpnn = BPNN.init(3,new int[]{2,2},3,0.023);
        //bpnn.initRandomWeight();
        ArrayList<double[][]> weights = new ArrayList<>();
        weights.add(new double[][]{{1, 1}, {1, 1}, {1, 1}});
        weights.add(new double[][]{{1, 1}, {1, 1}});
        weights.add(new double[][]{{1, 1, 1}, {1, 1, 1}});
        bpnn.setWeights(weights);

        //bpnn.initRandomWeight();

        bpnn.input(new double[][]{{1,1,2}});
        bpnn.formDiffuse();
        bpnn.printOut();

        bpnn.backPropagation(new double[][]{{22,10,18}});
//        System.out.println("输出");
//        bpnn.printOut();
//        System.out.println("差值");
//        bpnn.printDiffs();
    }
}
