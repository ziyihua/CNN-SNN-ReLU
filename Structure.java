import java.util.ArrayList;
import java.util.List;

/**
 * Created by ziyihua on 17/07/15.
 */
public class Structure {

    public static class network {
        public network() {
            layers = new ArrayList<LAYER>();
        }

        List<LAYER> layers;
        double[] ffb;
        double[][] ffW;
        double[] rL;
        double[] acc;
        double[][] fv;
        double[][] o;
        double[][] e;
        double L;
        double[][] od;
        double[][] fvd;
        double[][] dffW;
        double[] dffb;
        double first_layer_dropout;
    }

    public static class LAYER {
        public LAYER() {
            k = new ArrayList<weights>();
            a = new ArrayList<A>();
            d = new ArrayList<D>();
            dk = new ArrayList<DK>();
        }

        String type;
        int outmaps;
        int kernelsize;
        int scale;
        List<weights> k;
        double[] b;
        List<A> a;
        List<D> d;
        List<DK> dk;
        double[] db;
        int[] used_maps;
    }

    public static class weights {
        public weights() {
            k_list = new ArrayList();
        }

        ArrayList k_list;
    }

    public static class A {
        public A() {
            a_list = new ArrayList();
        }

        ArrayList a_list;
    }

    public static class D {
        public D() {
            d_list = new ArrayList();
        }

        ArrayList d_list;
    }

    public static class DK {
        public DK() {
            dk_list = new ArrayList();
        }

        ArrayList dk_list;
    }




    public static double[][][] flip3D(double[][][] input){
        double[][][] output = new double[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++) {
                    output[i][j][k]=input[input.length-1-i][input[0].length-1-j][input[0][0].length-1-k];
                }
            }
        }
        return output;
    }

}
