import org.apache.commons.math3.linear.Array2DRowRealMatrix;

public class TestMat {
    public static void main(String[] args) {
        double mat1[][] = new double[][]{{2,3,1}};
        double mat2[][] = new double[][]{{2,1},{2,1},{1,2}};
        Array2DRowRealMatrix matrix1 = new Array2DRowRealMatrix(mat1);
        Array2DRowRealMatrix matrix2 = new Array2DRowRealMatrix(mat2);
        Array2DRowRealMatrix multiply = matrix1.multiply(matrix2);
        System.out.println(multiply);
        System.out.println(multiply.getRow(0).length);
        System.out.println(multiply.getColumn(0).length);
        System.out.println(multiply.getEntry(0,0));
    }
}
