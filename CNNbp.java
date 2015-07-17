/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNbp extends Structure {

    public CNNbp(){
    }

    public static network CNNbp(String[][] architecture, network convnet, int[][] batch_y){
        int n = convnet.layers.size();

        //error
        double[][] e=new double[convnet.o.length][convnet.o[0].length];
        for (int i = 0; i < convnet.o.length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                e[i][j]=convnet.o[i][j]-(double) batch_y[i][j];
            }
        }
        convnet.e=e;
        //loss function
        double sum=0;
        for (int i = 0; i < convnet.o.length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                sum=sum+e[i][j]*e[i][j];
            }
        }
        double loss;
        loss = 0.5 * sum /convnet.o[0].length;
        convnet.L=loss;

        /**
         * backpropagation delta
         */
        //output delta od=e.*(o>0)
        double[][] od = new double[convnet.o.length][convnet.o[0].length];
        for (int i = 0; i < convnet.o.length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                if(convnet.o[i][j]>0) {
                    od[i][j] = convnet.e[i][j];
                }
                else {
                    od[i][j]=0;
                }
            }
        }
        convnet.od=od;


        //feature vector delta fvd=ffw'*od
        double[][] product = new double[convnet.ffW[0].length][convnet.o[0].length];
        for (int i = 0; i < convnet.ffW[0].length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                product[i][j] = 0;
            }
        }
        for (int i = 0; i < convnet.ffW[0].length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                for (int k = 0; k < convnet.o.length; k++) {
                    product[i][j]=product[i][j]+convnet.ffW[k][i]*convnet.od[k][j];
                }
            }
        }
        convnet.fvd = product;
        //only convolution layer has sigmoid function and if the last layer is convolution
        //fvd=fvd.*[fv.*(1-fv)]
        if ("c".equals(architecture[0][n-1])){
            double[][] dotprod_2 = new double[convnet.fvd.length][convnet.fvd[0].length];
            for (int i = 0; i < convnet.fvd.length; i++) {
                for (int j = 0; j < convnet.fvd[0].length; j++) {
                    dotprod_2[i][j]=convnet.fv[i][j]*(1-convnet.fv[i][j]);
                }
            }
            double[][] fvd = new double[convnet.fvd.length][convnet.fvd[0].length];
            for (int i = 0; i < convnet.fvd.length; i++) {
                for (int j = 0; j < convnet.fvd[0].length; j++) {
                    fvd[i][j]=convnet.fvd[i][j]*dotprod_2[i][j];
                }
            }
            convnet.fvd=fvd;
        }

        /**
         * reshape feature vector deltas into output map style
         */
        LAYER layer_last = convnet.layers.get(n - 1);

        int a = ((double[][][]) layer_last.a.get(0).a_list.get(0)).length;
        int b = ((double[][][]) layer_last.a.get(0).a_list.get(0))[0].length;
        int c = ((double[][][]) layer_last.a.get(0).a_list.get(0))[0][0].length;
        int outputmaps = layer_last.a.get(0).a_list.size();
        D Dl = new D();

        if (convnet.layers.get(n-1).d.isEmpty()) {
            convnet.layers.get(n-1).d.add(0, Dl);
        }else convnet.layers.get(n-1).d.set(0,Dl);

        for (int i = 0; i < outputmaps; i++) {
            double[][][] d = new double[a][b][c];
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < b; k++) {
                    for (int l = 0; l < a; l++) {
                        d[l][k][j]=convnet.fvd[i*a*b+k*a+l][j];
                    }
                }
            }
            convnet.layers.get(n-1).d.get(0).d_list.add(i,d);
        }

        for (int i = n-2; i >=0 ; i--) {
            if("c".equals(architecture[0][i])){

                D Dc = new D();
                if(convnet.layers.get(i).d.isEmpty()){
                    convnet.layers.get(i).d.add(0,Dc);
                }else convnet.layers.get(i).d.set(0,Dc);

                int num_a = convnet.layers.get(i).a.get(0).a_list.size();
                int size_a1=((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(0)).length;
                int size_a2=((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(0))[0].length;
                int size_a3=((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(0))[0][0].length;

                //assigned to unused maps
                double[][][] zeros = new double[size_a1][size_a2][size_a3];
                for (int k = 0; k < size_a1; k++) {
                    for (int l = 0; l < size_a2; l++) {
                        for (int m = 0; m < size_a3; m++) {
                            zeros[k][l][m]=0.0;
                        }
                    }
                }

                for (int j = 0; j < num_a; j++) {
                    if (convnet.layers.get(i).used_maps[j]==1) {
                        //expand the size of feature deltas to that of the current layer (replicate each row and column n times where n is the scale of the next subsampling layer)
                        int sub_scale = convnet.layers.get(i + 1).scale;
                        double[][][] expand = new double[size_a1][size_a2][size_a3];
                        for (int k = 0; k < size_a3; k++) {
                            for (int l = 0; l < size_a2; l++) {
                                for (int m = 0; m < size_a1; m++) {
                                    expand[m][l][k] = ((double[][][]) convnet.layers.get(i + 1).d.get(0).d_list.get(j))[m / sub_scale][l / sub_scale][k];
                                }
                            }
                        }
                        //d_current_layer=ReLU'(a.*expand(d_previous_layer)/scale^2)
                        double[][][] product_1 = new double[size_a1][size_a2][size_a3];
                        for (int k = 0; k < size_a3; k++) {
                            for (int l = 0; l < size_a2; l++) {
                                for (int m = 0; m < size_a1; m++) {
                                    product_1[m][l][k] = ((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(j))[m][l][k] * expand[m][l][k]/((double) sub_scale * sub_scale);
                                }
                            }
                        }

                        double[][][] derivative = new double[size_a1][size_a2][size_a3];
                        for (int k = 0; k < size_a3; k++) {
                            for (int l = 0; l < size_a2; l++) {
                                for (int m = 0; m < size_a1; m++) {
                                    if (product_1[m][l][k]>0) {
                                        derivative[m][l][k] = 1.0;
                                    }
                                    else {
                                        derivative[m][l][k]=0.0;
                                    }
                                }
                            }
                        }
                        convnet.layers.get(i).d.get(0).d_list.add(j, derivative);
                    }
                    else {
                        convnet.layers.get(i).d.get(0).d_list.add(j, zeros);
                    }
                }
            }else if ("s".equals(architecture[0][i])){

                D Ds = new D();
                if(convnet.layers.get(i).d.isEmpty()){
                    convnet.layers.get(i).d.add(0,Ds);
                }else convnet.layers.get(i).d.set(0,Ds);

                int num_a = convnet.layers.get(i).a.get(0).a_list.size();
                int size_a1=((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(0)).length;
                int size_a2=((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(0))[0].length;
                int size_a3=((double[][][]) convnet.layers.get(i).a.get(0).a_list.get(0))[0][0].length;

                for (int j = 0; j < num_a; j++){

                    double[][][] z = new double[size_a1][size_a2][size_a3];
                    for (int k = 0; k < size_a1; k++) {
                        for (int l = 0; l < size_a2; l++) {
                            for (int m = 0; m < size_a3; m++) {
                                z[k][l][m]=0;
                            }
                        }
                    }
                    //reverse-convolution
                    for (int k = 0; k < convnet.layers.get(i+1).a.get(0).a_list.size(); k++) {
                        //matrix-wise convolution along 3rd dimension
                        for (int l = 0; l < size_a3; l++) {
                            int size_d1 = ((double[][][]) convnet.layers.get(i+1).d.get(0).d_list.get(j)).length;
                            int size_d2 = ((double[][][]) convnet.layers.get(i+1).d.get(0).d_list.get(j))[0].length;
                            double[][] d_one = new double[size_d1][size_d2];
                            for (int m = 0; m < size_d2; m++) {
                                for (int o = 0; o < size_d1; o++) {
                                    d_one[o][m] = ((double[][][]) convnet.layers.get(i+1).d.get(0).d_list.get(k))[o][m][l];
                                }
                            }
                            double[][] z_conv;
                            z_conv = Convolution.convolution2D_full_wof(d_one, size_d1, size_d2, (double[][]) convnet.layers.get(i + 1).k.get(0).k_list.get(k + j * convnet.layers.get(i+1).outmaps),convnet.layers.get(i+1).kernelsize,convnet.layers.get(i+1).kernelsize);
                            for (int m = 0; m < size_a2; m++) {
                                for (int o = 0; o < size_a1; o++) {
                                    z[o][m][l]=z[o][m][l]+z_conv[o][m];
                                }
                            }
                        }
                    }
                    convnet.layers.get(i).d.get(0).d_list.add(j,z);
                }
            }
        }

        /**
         * Calculate gradients
         */
        for (int i = 1; i < n; i++) {
            if("c".equals(architecture[0][i])) {

                int num_a_current = convnet.layers.get(i).a.get(0).a_list.size();
                int num_a_previous = convnet.layers.get(i - 1).a.get(0).a_list.size();

                DK DKc = new DK();
                if (convnet.layers.get(i).dk.isEmpty()) {
                    convnet.layers.get(i).dk.add(0, DKc);
                } else convnet.layers.get(i).dk.set(0, DKc);

                for (int k = 0; k < num_a_previous; k++) {
                    for (int j = 0; j < num_a_current; j++) {
                        //gradient dk is calculated as convolution of a in the last layer with k in the current layer
                        double[][][] flipall = flip3D(((double[][][]) convnet.layers.get(i - 1).a.get(0).a_list.get(k)));
                        double[][][] kernel = (double[][][]) convnet.layers.get(i).d.get(0).d_list.get(j);
                        double[][] conv = Convolution.convolution3D(flipall, flipall.length, flipall[0].length, flipall[0][0].length, kernel, kernel.length, kernel[0].length, kernel[0][0].length);
                        double[][] dk = new double[conv.length][conv[0].length];
                        for (int l = 0; l < conv.length; l++) {
                            for (int m = 0; m < conv[0].length; m++) {
                                dk[l][m] = conv[l][m] / (double) kernel[0][0].length;
                            }
                        }
                        convnet.layers.get(i).dk.get(0).dk_list.add(j + k * convnet.layers.get(i).outmaps, dk);
                    }
                }

                //db=sum(d)/size
                double[] d_sum_array = new double[num_a_current];
                for (int j = 0; j < num_a_current; j++) {
                    double d_sum = 0;
                    int size_d1 = ((double[][][]) convnet.layers.get(i).d.get(0).d_list.get(j)).length;
                    int size_d2 = ((double[][][]) convnet.layers.get(i).d.get(0).d_list.get(j))[0].length;
                    int size_d3 = ((double[][][]) convnet.layers.get(i).d.get(0).d_list.get(j))[0][0].length;
                    for (int k = 0; k < size_d1; k++) {
                        for (int l = 0; l < size_d2; l++) {
                            for (int m = 0; m < size_d3; m++) {
                                d_sum = d_sum + ((double[][][]) convnet.layers.get(i).d.get(0).d_list.get(j))[k][l][m];
                            }
                        }
                    }
                    d_sum_array[j] = d_sum / (double) size_d3;
                }
                convnet.layers.get(i).db=d_sum_array;
            }
        }

        //dffW=od*fv'/size
        double[][] od_product = new double[convnet.od.length][convnet.fv.length];
        for (int i = 0; i < convnet.od.length; i++) {
            for (int j = 0; j < convnet.fv.length; j++) {
                od_product[i][j]=0;
            }
        }

        for (int row = 0; row < od_product.length; row++) {
            for (int i = 0; i < od_product[0].length; i++) {
                for (int j = 0; j < convnet.fv[0].length; j++) {
                    od_product[row][i]=od_product[row][i]+convnet.od[row][j]*convnet.fv[i][j];
                }
            }
        }

        for (int i = 0; i < od_product.length; i++) {
            for (int j = 0; j < od_product[0].length; j++) {
                od_product[i][j]=od_product[i][j]/(double) convnet.od[0].length;
            }
        }
        convnet.dffW=od_product;

        //dffb=0
        double[] dffb = new double[convnet.od.length];
        for (int i = 0; i < convnet.od.length; i++) {
            dffb[i]=0;
        }
        convnet.dffb=dffb;

        return convnet;
    }
}
