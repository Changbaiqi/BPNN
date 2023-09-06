import com.bpnn.BPNN;

import java.util.ArrayList;

public class Test2 {
    public static void main(String[] args) {
        BPNN bpnn = BPNN.init(28*28,new int[]{16,16},10,0.025);
        bpnn.setWeights(BPNN.readWeights("E:/IntelliJ_WorkSpace/BPNNImage", "weightsTest53.txt"));
        ArrayList<double[][]> weights = bpnn.getWeights();
        for (int y = 0; y < weights.get(0).length; y++) {
            for (int x = 0; x < weights.get(0)[y].length; x++) {
                System.out.print(weights.get(0)[y][x]+" ");
            }
            System.out.println();
        }
    }
}
