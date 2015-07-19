import java.util.ArrayList;
import java.util.Random;

/**
 * Created by ziyihua on 18/07/15.
 */
public class Convlifsim extends Structure {
    public Convlifsim(){
    }


    public static network Convlifsim(network convnet, double[][][] test_x, int[][] test_y_new, int[] test_y, double t_ref, double threshold, double dt, double duration, double report_every, double max_rate){

        int num_examples = test_x[0][0].length;
        int num_classes = test_y_new.length;

        Random r = new Random();

        //initialization
        for (int i = 0; i < convnet.layers.size(); i++) {
            M mem = new M();
            convnet.layers.get(i).m.add(0,mem);
            R refrac_end = new R();
            convnet.layers.get(i).r.add(0,refrac_end);
            S sum_spikes = new S();
            convnet.layers.get(i).s.add(0,sum_spikes);
            int outputmaps = convnet.layers.get(i).a.get(0).a_list.size();
            for (int j = 0; j < outputmaps; j++) {
                double[][][] correctly_sized_zeros = new double[((double[][][])convnet.layers.get(i).a.get(0).a_list.get(j)).length][((double[][][])convnet.layers.get(i).a.get(0).a_list.get(j))[0].length][num_examples];
                for (int k = 0; k < correctly_sized_zeros.length; k++) {
                    for (int l = 0; l < correctly_sized_zeros[0].length; l++) {
                        for (int m = 0; m < correctly_sized_zeros[0][0].length; m++) {
                            correctly_sized_zeros[k][l][m]=0;
                        }
                    }
                }
                convnet.layers.get(i).m.get(0).m_list.add(j,correctly_sized_zeros);
                convnet.layers.get(i).r.get(0).r_list.add(j,correctly_sized_zeros);
                convnet.layers.get(i).s.get(0).s_list.add(j,correctly_sized_zeros);
            }
        }

        double[][] sum_fv = new double[convnet.ffW[0].length][num_examples];
        for (int i = 0; i < sum_fv.length; i++) {
            for (int j = 0; j < sum_fv[0].length; j++) {
                sum_fv[i][j]=0;
            }
        }
        convnet.sum_fv=sum_fv;

        double[][] omem = new double[num_classes][num_examples];
        for (int i = 0; i < omem.length; i++) {
            for (int j = 0; j < omem[0].length; j++) {
                omem[i][j]=0;
            }
        }
        convnet.omem=omem;

        double[][] o_refrac_end = new double[num_classes][num_examples];
        for (int i = 0; i < o_refrac_end.length; i++) {
            for (int j = 0; j < o_refrac_end[0].length; j++) {
                o_refrac_end[i][j]=0;
            }
        }
        convnet.o_refrac_end=o_refrac_end;

        double[][] o_sum_spikes = new double[num_classes][num_examples];
        for (int i = 0; i < o_sum_spikes.length; i++) {
            for (int j = 0; j < o_sum_spikes[0].length; j++) {
                o_sum_spikes[i][j]=0;
            }
        }
        convnet.o_sum_spikes=o_sum_spikes;

        double[] performance = new double[(int)(duration/dt)];


        for (double i = 0; i <= duration ; i=i+dt) {
            //create Poisson distributed spikes form the input images (for all images in parallel)
            double rescale_fac = 1/(dt*max_rate);
            double[][][] spike_snapshot = new double[test_x.length][test_x[0].length][test_x[0][0].length];
            for (int j = 0; j < spike_snapshot.length; j++) {
                for (int k = 0; k < spike_snapshot[0].length; k++) {
                    for (int l = 0; l < spike_snapshot[0][0].length; l++) {
                        double RandomValue = r.nextDouble();
                        spike_snapshot[j][k][l]= RandomValue * rescale_fac;
                    }
                }
            }
            int[][][] inp_image = new int[test_x.length][test_x[0].length][test_x[0][0].length];
            for (int j = 0; j < inp_image.length; j++) {
                for (int k = 0; k < inp_image[0].length; k++) {
                    for (int l = 0; l < inp_image[0][0].length; l++) {
                        if (spike_snapshot[j][k][l] <= test_x[j][k][l])
                            inp_image[j][k][l] = 1;
                        else inp_image[j][k][l] = 0;
                    }
                }
            }

            S spikes = new S();
            if(convnet.layers.get(0).s.isEmpty()){
                convnet.layers.get(0).s.add(0, spikes);
            }else convnet.layers.get(0).s.set(0,spikes);
            convnet.layers.get(0).s.get(0).s_list.add(0,inp_image);

            ArrayList mem_i = new ArrayList();
            double[][][] mem_i_m = new double[inp_image.length][inp_image[0].length][inp_image[0][0].length];
            for (int j = 0; j < mem_i_m.length; j++) {
                for (int k = 0; k < mem_i_m[0].length; k++) {
                    for (int l = 0; l < mem_i_m[0][0].length; l++) {
                        mem_i_m[j][k][l]=((double[][][])convnet.layers.get(0).m.get(0).m_list.get(0))[j][k][l]+inp_image[j][k][l];
                    }
                }
            }
            mem_i.add(0,mem_i_m);
            convnet.layers.get(0).m.get(0).m_list.clear();
            convnet.layers.get(0).m.get(0).m_list.addAll(mem_i);

            ArrayList sum_spikes_i = new ArrayList();
            double[][][] sum_spikes_i_m = new double[inp_image.length][inp_image[0].length][inp_image[0][0].length];
            for (int j = 0; j < inp_image.length; j++) {
                for (int k = 0; k < inp_image[0].length; k++) {
                    for (int l = 0; l < inp_image[0][0].length; l++) {
                        sum_spikes_i_m[j][k][l]=((double[][][])convnet.layers.get(0).s.get(0).s_list.get(0))[j][k][l]+inp_image[j][k][l];
                    }
                }
            }
            sum_spikes_i.add(0,sum_spikes_i_m);
            convnet.layers.get(0).s.get(0).s_list.clear();
            convnet.layers.get(0).s.get(0).s_list.addAll(sum_spikes_i);

            int inputmaps = 1;





        }

        return convnet;
    }
}
