using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;

public class xor : MonoBehaviour
{


    [DllImport("5a_AL1_firstDLL")]
    static extern System.IntPtr createNetworkModel(int[] nbDim, int nbLayer);

    [DllImport("5a_AL1_firstDLL")]
    static extern void trainMLP(double[] exemples, int nbExemples, int[] expectedResult, System.IntPtr network, int[] neurones, int nbLayer, int maxIteration, double alpha);

    [DllImport("5a_AL1_firstDLL")]
    static extern System.IntPtr classifyMLP(System.IntPtr network, double[] element, int[] neuronsCount, int nbLayer);

    [DllImport("5a_AL1_firstDLL")]
    static extern void remove2DModel(System.IntPtr model, int nbLayer);

    [DllImport("5a_AL1_firstDLL")]
    static extern void removeNetworkModel(System.IntPtr model, int[] nbDim, int nbLayer);

    [DllImport("5a_AL1_firstDLL")]
    static extern void removeLinearModel(System.IntPtr model);

    [SerializeField]
    Transform[] planTargetTransforms;

    [SerializeField]
    Transform[] targetTransforms;

    [SerializeField]
    MeshRenderer[] targetMeshRenderers;

    [SerializeField]
    Material redMat;

    [SerializeField]
    Material blueMat;

    [SerializeField]
    Material greenMat;

    class ClassificationTraining
    {
        public int NbElements { get; set; }

        public int NbInputs { get; set; }

        public double[] TrainingExemples { get; set; }

        public int[] ExpectedResults { get; set; }

        public int MaxIteration { get; set; }
    }
    class RegressionTraining2
    {
        public int NbElements { get; set; }

        public int NbInputs { get; set; }

        public double[] TrainingExemples { get; set; }

        public double[] ExpectedResults { get; set; }

        public int MaxIteration { get; set; }
    }

    ClassificationTraining createTrainingConfiguration()
    {
        ClassificationTraining c = new ClassificationTraining();
        c.NbElements = targetTransforms.Length;
        c.NbInputs = 2;

        c.TrainingExemples = new double[c.NbElements * c.NbInputs];
        c.ExpectedResults = new int[c.NbElements];

        for (int i = 0; i < targetMeshRenderers.Length; ++i)
        {
            MeshRenderer mr = targetMeshRenderers[i].GetComponent<MeshRenderer>();
            if (mr.material.color == redMat.color)
            {
                Transform t = targetTransforms[i].GetComponent<Transform>();
                c.TrainingExemples[i * c.NbInputs] = t.position.x;
                c.TrainingExemples[i * c.NbInputs + 1] = t.position.z;
                c.ExpectedResults[i] = 1;
            }
            if (mr.material.color == blueMat.color)
            {
                Transform t = targetTransforms[i].GetComponent<Transform>();
                c.TrainingExemples[i * c.NbInputs] = t.position.x;
                c.TrainingExemples[i * c.NbInputs + 1] = t.position.z;
                c.ExpectedResults[i] = -1;
            }
        }

        c.MaxIteration = 100000;

        return c;
    }

    void ClassificationScenario(ClassificationTraining c, System.IntPtr model, int[] inputneuro)
    {
        //printModel(model);

        trainMLP(c.TrainingExemples, c.NbElements, c.ExpectedResults, model, inputneuro, inputneuro.Length, c.MaxIteration, 0.01);

        //printModel(model);

        adjustPlan(model, c.NbInputs, inputneuro);

        //removeModel(model);
    }

    void adjustPlan(System.IntPtr ptrModel, int nbInput, int[] inputneuro)
    {
        for (int i = 0; i < planTargetTransforms.Length; i++)
        {
            Transform t = planTargetTransforms[i].GetComponent<Transform>();
            MeshRenderer mr = planTargetTransforms[i].GetComponent<MeshRenderer>();

            double[] input = new double[2];
            input[0] = t.position.x;
            input[1] = t.position.z;

            System.IntPtr output = classifyMLP(ptrModel, input, inputneuro, inputneuro.Length);



            int[] managedOutput = new int[1];
            Marshal.Copy(output, managedOutput, 0, 1);
            switch (managedOutput[0])
            {
                case 1:
                    mr.material.color = redMat.color;
                    t.position += Vector3.up * 1;
                    break;
                case -1:
                    mr.material.color = blueMat.color;
                    break;
            }
            //t.position += Vector3.up * 1;
        }
    }
    // Use this for initialization
    void Start()
    {
        var c = createTrainingConfiguration();
        int[] input = new int[] { 2, 3, 1 };
        System.IntPtr model = createNetworkModel(input, input.Length);

        ClassificationScenario(c, model, input);

        removeNetworkModel(model,input, input.Length);
        removeLinearModel(model);
    }

    // Update is called once per frame
    void Update()
    {

    }
}
