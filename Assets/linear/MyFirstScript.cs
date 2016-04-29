using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;

public class MyFirstScript : MonoBehaviour {

    [DllImport("5a_AL1_firstDLL")] 
    static extern public int Hello42();

    [DllImport("5a_AL1_firstDLL")]
    static extern public void removeModel(System.IntPtr model);

    [DllImport("5a_AL1_firstDLL")]
    static extern public System.IntPtr createModelLinear(int nbInput);

    [DllImport("5a_AL1_firstDLL")]
    static extern public void trainPLA(double[] exemples, int nbInput, int nbExemples, int[] expectedResult, System.IntPtr model, int maxIteration);

    [DllImport("5a_AL1_firstDLL")]
    static extern public void perceptron_regression(System.IntPtr model, double[] inputs, int nbInput, int nbCount, double[] expectedResult);


    [DllImport("5a_AL1_firstDLL")]
    static extern public int classifyLinear(System.IntPtr model, double[] element, int nbInput);

    [DllImport("5a_AL1_firstDLL")]
    static extern public double classifyRegression(System.IntPtr model, double[] element, int nbInput);


    [DllImport("5a_AL1_firstDLL")]
    static extern public void trainRosenblatt(double[] exemples, int nbInput, int nbExemples, int[] expectedResult, System.IntPtr model, int maxIteration);

    [DllImport("5a_AL1_firstDLL")]
    static extern public int classifyRosenblatt(System.IntPtr model, double[] element, int nbInput);

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

    class LinearTraining
    {
        public int NbElements { get; set; }

        public int NbInputs { get; set; }

        public double[] TrainingExemples { get; set; }

        public int[] ExpectedResults { get; set; }

        public int MaxIteration { get; set; }
    }
    class LinearTraining2
    {
        public int NbElements { get; set; }

        public int NbInputs { get; set; }

        public double[] TrainingExemples { get; set; }

        public double[] ExpectedResults { get; set; }

        public int MaxIteration { get; set; }
    }

    LinearTraining createTrainingConfiguration()
    {
        LinearTraining c = new LinearTraining();
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
                c.TrainingExemples[i * c.NbInputs ] = t.position.x;
                c.TrainingExemples[i * c.NbInputs + 1] = t.position.z;
                c.ExpectedResults[i] = -1;
            }
            if (mr.material.color == blueMat.color)
            {
                Transform t = targetTransforms[i].GetComponent<Transform>();
                c.TrainingExemples[i * c.NbInputs ] = t.position.x;
                c.TrainingExemples[i * c.NbInputs +1] = t.position.z;
                c.ExpectedResults[i] = 1;
            }            
        }

        c.MaxIteration = 10000;

        return c;
    }
    LinearTraining2 createTrainingConfiguration2()
    {
        LinearTraining2 c = new LinearTraining2();
        c.NbElements = targetTransforms.Length;
        c.NbInputs = 2;

        c.TrainingExemples = new double[c.NbElements * c.NbInputs];
        c.ExpectedResults = new double[c.NbElements];

        for (int i = 0; i < targetMeshRenderers.Length; ++i)
        {
            MeshRenderer mr = targetMeshRenderers[i].GetComponent<MeshRenderer>();
            if (mr.material.color == redMat.color)
            {
                Transform t = targetTransforms[i].GetComponent<Transform>();
                c.TrainingExemples[i * c.NbInputs] = t.position.x;
                c.TrainingExemples[i * c.NbInputs + 1] = t.position.z;
                c.ExpectedResults[i] = 1.0;
            }
            if (mr.material.color == blueMat.color)
            {
                Transform t = targetTransforms[i].GetComponent<Transform>();
                c.TrainingExemples[i * c.NbInputs] = t.position.x;
                c.TrainingExemples[i * c.NbInputs + 1] = t.position.z;
                c.ExpectedResults[i] = -1.0;
            }
        }

        c.MaxIteration = 10000;

        return c;
    }

    void linearScenario(LinearTraining c,System.IntPtr model)
    {
        //printModel(model);

        trainPLA(c.TrainingExemples, c.NbInputs, c.NbElements, c.ExpectedResults, model, c.MaxIteration);

        //printModel(model);
        
        adjustPlan(model, c.NbInputs);

        removeModel(model);
    }
    void linearScenarioRegression(LinearTraining2 c,System.IntPtr model)
    {        
        perceptron_regression(model, c.TrainingExemples, c.NbElements,c.NbInputs, c.ExpectedResults);

        adjustPlan2(model);

        removeModel(model);
    }

    void adjustPlan(System.IntPtr ptrModel, int nbInput)
    {
        for (int i = 0; i < planTargetTransforms.Length; i++)
        {
            Transform t = planTargetTransforms[i].GetComponent<Transform>();

            double[] input = new double[2];
            input[0] = t.position.x;
            input[1] = t.position.z;

            if(classifyLinear(ptrModel, input, 2) == 1 )
                t.position += Vector3.up * 1;
        }
    }
    void adjustPlan2(System.IntPtr ptrModel)
    {

        for (int i = 0; i < planTargetTransforms.Length; i++)
        {
            Transform t = planTargetTransforms[i].GetComponent<Transform>();
            MeshRenderer mr = planTargetTransforms[i].GetComponent<MeshRenderer>();

            double[] input = new double[2];
            input[0] = t.position.x;
            input[1] = t.position.z;
            double val = classifyRegression(ptrModel, input, 2);            
            val = (val + 1) / 2;

            mr.material.color = new Color((float)val, 0f, (float)(1.0 - val));
           
        }
    }

    void printModel(System.IntPtr model)
    {
        for (int i = 0; i < planTargetTransforms.Length; i++)
        {
            Transform t = planTargetTransforms[i].GetComponent<Transform>();

            double []input = new double[2];
            input[0] = t.position.x;
            input[1] = t.position.z;

            Debug.Log(classifyLinear(model, input, 2));
        }
    
    }


    // Use this for initialization
    void Start () {
        Debug.Log("Hello les 5A AL1");
        Debug.Log("From DLL  ==>  : " + Hello42());
        var c = createTrainingConfiguration();

        System.IntPtr model = createModelLinear(c.NbInputs);

        linearScenario(c,model);
        model = createModelLinear(c.NbInputs+1);
        var c2 = createTrainingConfiguration2();

        linearScenarioRegression(c2,model);
    }
	
	// Update is called once per frame
	void Update () {
	
	}
}
