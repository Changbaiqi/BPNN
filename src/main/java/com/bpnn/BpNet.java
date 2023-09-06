package com.bpnn;

public  class BpNet
{
    private  static  final  int IM =  1;                   //输入层数量
    private  static  final  int RM =  8;                   //隐含层数量
    private  static  final  int OM =  1;                   //输出层数量
    private  double learnRate =  0.55;                        //学习速率
    private  double alfa =  0.67;                             //动量因子
    private  double Win[][] =  new  double[IM][RM];      //输入到隐含连接权值
    private  double oldWin[][] =  new  double[IM][RM];
    private  double old1Win[][] =  new  double[IM][RM];
    private  double dWin[][] =  new  double[IM][RM];
    private  double Wout[][] =  new  double[RM][OM];     //隐含到输出连接权值
    private  double oldWout[][] =  new  double[RM][OM];
    private  double old1Wout[][] =  new  double[RM][OM];
    private  double dWout[][] =  new  double[RM][OM];
    private  double Xi[] =  new  double[IM];
    private  double Xj[] =  new  double[RM];
    private  double XjActive[] =  new  double[RM];
    private  double Xk[] =  new  double[OM];
    private  double Ek[] =  new  double[OM];
    private  double J =  0.1;

    public  void train()
    {
        double y;
        int n =  0;
        //初始化权值和清零//
        bpNetinit();
        System.out.println( "training...");
        while(J > Math.pow( 10, - 17))
        {
            for(n =  0; n <  20; n++)
            {
                y = n *  2 +  23;  //逼近对象
                //前向计算输出过程//
                bpNetForwardProcess(n /  100.0, y /  100.0);
                //反向学习修改权值//
                bpNetReturnProcess();
            }
        }
        //在线学习后输出//
        for(n =  0; n <  20; n++)
        {
            y = n *  2 +  23;     //逼近对象
            System.out.printf( "%.1f  ", y);
            System.out.printf( "%f  ", bpNetOut(n /  100.0)[ 0] *  100.0);
            System.out.println( "J=" + J);
        }
        System.out.println( "n=20 " +  "Out:" +  this.bpNetOut( 20 /  100.0)[ 0] *  100);
    }
    //
    // BP神经网络权值随机初始化
    // Win[i][j]和Wout[j][k]权值初始化为[-0.5,0.5]之间
    //
    public  void bpNetinit()
    {
        //初始化权值和清零//
        for( int i =  0; i < IM; i++)
            for( int j =  0; j < RM; j++)
            {
                Win[i][j] =  0.5 - Math.random();
                Xj[j] =  0;
            }
        for( int j =  0; j < RM; j++)
            for( int k =  0; k < OM; k++)
            {
                Wout[j][k] =  0.5 - Math.random();
                Xk[k] =  0;
            }
    }
    //
    // BP神经网络前向计算输出过程
    // @param inputParameter 归一化后的理想输入值(单个double值)
    // @param outputParameter  归一化后的理想输出值(单个double值)
    //
    public  void bpNetForwardProcess( double inputParameter,  double outputParameter)
    {
        double input[] = {inputParameter};
        double output[] = {outputParameter};
        bpNetForwardProcess(input, output);
    }
    //
    // BP神经网络前向计算输出过程--多个输入，多个输出
    // @param inputParameter  归一化后的理想输入数组值
    // @param outputParameter  归一化后的理想输出数组值
    //
    public  void bpNetForwardProcess( double inputParameter[],  double outputParameter[])
    {
        for( int i =  0; i < IM; i++)
        {
            Xi[i] = inputParameter[i];
        }
        //隐含层权值和计算//
        for( int j =  0; j < RM; j++)
        {
            Xj[j] =  0;
            for( int i =  0; i < IM; i++)
            {
                Xj[j] = Xj[j] + Xi[i] * Win[i][j];
            }
        }
        //隐含层S激活输出//
        for( int j =  0; j < RM; j++)
        {
            XjActive[j] =  1 / ( 1 + Math.exp(-Xj[j]));
        }
        //输出层权值和计算//
        for( int k =  0; k < OM; k++)
        {
            Xk[k] =  0;
            for( int j =  0; j < RM; j++)
            {
                Xk[k] = Xk[k] + XjActive[j] * Wout[j][k];
            }
        }
        //计算输出与理想输出的偏差//
        for( int k =  0; k < OM; k++)
        {
            Ek[k] = outputParameter[k] - Xk[k];
        }
        //误差性能指标//
        J =  0;
        for( int k =  0; k < OM; k++)
        {
            J = J + Ek[k] * Ek[k] /  2.0;
        }
    }
    //
    //BP神经网络反向学习修改连接权值过程
    //
    public  void bpNetReturnProcess()
    {
        //反向学习修改权值//
        for( int i =  0; i < IM; i++)  //输入到隐含权值修正
        {
            for( int j =  0; j < RM; j++)
            {
                for( int k =  0; k < OM; k++)
                {
                    dWin[i][j] = dWin[i][j] + learnRate * (Ek[k] * Wout[j][k] * XjActive[j] * ( 1 - XjActive[j]) * Xi[i]);
                }
                Win[i][j] = Win[i][j] + dWin[i][j] + alfa * (oldWin[i][j] - old1Win[i][j]);

                old1Win[i][j] = oldWin[i][j];
                oldWin[i][j] = Win[i][j];
            }
        }

        for( int j =  0; j < RM; j++)  //隐含到输出权值修正
        {
            for( int k =  0; k < OM; k++)
            {
                dWout[j][k] = learnRate * Ek[k] * XjActive[j];
                Wout[j][k] = Wout[j][k] + dWout[j][k] + alfa * (oldWout[j][k] - old1Wout[j][k]);

                old1Wout[j][k] = oldWout[j][k];
                oldWout[j][k] = Wout[j][k];
            }
        }
    }
    //
    // BP神经网络前向计算输出，训练结束后测试输出
    // @param inputParameter  测试的归一化后的输入值
    // @return  返回归一化后的BP神经网络输出值，需逆归一化
    //
    public  double[] bpNetOut( double inputParameter)
    {
        double[] input = {inputParameter};
        return bpNetOut(input);
    }
    //
    // BP神经网络前向计算输出，训练结束后测试输出
    // @param inputParameter 测试的归一化后的输入数组
    // @return  返回归一化后的BP神经网络输出数组
    //
    public  double[] bpNetOut( double[] inputParameter)
    {
        //在线学习后输出//
        for( int i =  0; i < IM; i++)
        {
            Xi[i] = inputParameter[i];
        }
        //隐含层权值和计算//
        for( int j =  0; j < RM; j++)
        {
            Xj[j] =  0;
            for( int i =  0; i < IM; i++)
            {
                Xj[j] = Xj[j] + Xi[i] * Win[i][j];
            }
        }
        //隐含层S激活输出//
        for( int j =  0; j < RM; j++)
        {
            XjActive[j] =  1 / ( 1 + Math.exp(-Xj[j]));
        }
        //输出层权值和计算//
        double Uk[] =  new  double[OM];
        for( int k =  0; k < OM; k++)
        {
            Xk[k] =  0;
            for( int j =  0; j < RM; j++)
            {
                Xk[k] = Xk[k] + XjActive[j] * Wout[j][k];
                Uk[k] = Xk[k];
            }
        }
        return Uk;
    }
}