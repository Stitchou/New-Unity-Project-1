using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;

public class Mycs : MonoBehaviour {


    [DllImport("5a_AL1_firstDLL")]
    static extern System.IntPtr createNetworkModel(int[] nbDim, int nbLayer, int nbInput);

    [DllImport("5a_AL1_firstDLL")]
    static extern void trainMLP(double[] exemples, int nbInput, int nbExemples, int[] expectedResult, System.IntPtr network, int[] neurones, int nbLayer, int maxIteration);

    [DllImport("5a_AL1_firstDLL")]
    static extern System.IntPtr classifyMLP(System.IntPtr network, double[] element, int nbInput, int[] neuronsCount, int nbLayer);

    [DllImport("5a_AL1_firstDLL")]
    static extern void remove2DModel(System.IntPtr model, int nbLayer);

    [DllImport("5a_AL1_firstDLL")]
    static extern void removeNetworkModel(System.IntPtr model, double[] nbDim, int nbLayer);

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
                c.ExpectedResults[i] = -1;
            }
            if (mr.material.color == blueMat.color)
            {
                Transform t = targetTransforms[i].GetComponent<Transform>();
                c.TrainingExemples[i * c.NbInputs] = t.position.x;
                c.TrainingExemples[i * c.NbInputs + 1] = t.position.z;
                c.ExpectedResults[i] = 1;
            }
        }

        c.MaxIteration = 10000;

        return c;
    }

    void ClassificationScenario(ClassificationTraining c, System.IntPtr model)
    {
        //printModel(model);
        int[] inputneuro = new int[2];
        inputneuro[0] = 2;
        inputneuro[1] = 1;
        trainMLP(c.TrainingExemples, c.NbInputs, c.NbElements, c.ExpectedResults, model,inputneuro,2, c.MaxIteration);

        //printModel(model);

        adjustPlan(model, c.NbInputs);

        //removeModel(model);
    }

    void adjustPlan(System.IntPtr ptrModel, int nbInput)
    {
        for (int i = 0; i < planTargetTransforms.Length; i++)
        {
            Transform t = planTargetTransforms[i].GetComponent<Transform>();
            MeshRenderer mr = planTargetTransforms[i].GetComponent<MeshRenderer>();

            double[] input = new double[2];
            input[0] = t.position.x;
            input[1] = t.position.z;
            int[] inputneuro = new int[2];
            inputneuro[0] = 2;
            inputneuro[1] = 1;


            System.IntPtr output = classifyMLP(ptrModel, input, 2, inputneuro, 2);

            

            int[] managedOutput = new int[1];
            Marshal.Copy(output, managedOutput, 0, 1);
            switch(managedOutput[0])
            {
                case -1:
                    mr.material.color = redMat.color;
                    break;
                case 1:
                    mr.material.color = blueMat.color;
                    break; 
            }
            //t.position += Vector3.up * 1;
        }
    }
    // Use this for initialization
    void Start () {
        var c = createTrainingConfiguration();
        int[] input = new int[2];
        input[0] = 2;
        input[1] = 1;
        System.IntPtr model = createNetworkModel(input,2,c.NbInputs);

        ClassificationScenario(c, model);
    }
	
	// Update is called once per frame
	void Update () {
	
	}
}
