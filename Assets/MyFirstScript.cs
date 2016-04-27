using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;

public class MyFirstScript : MonoBehaviour {

    [DllImport("5a_AL1_firstDLL")] 
    static extern public int Hello42();

    [SerializeField]
    Transform[] targetsTransform;

    [SerializeField]
    Material redMat;

    [SerializeField]
    Material blueMat;

    [SerializeField]
    Material greenMat;

    [SerializeField]
    MeshRenderer targetMeshRenderer;

    // Use this for initialization
    void Start () {
        Debug.Log("Hello les 5A AL1");
        Debug.Log("From DLL  ==>  : " + Hello42());

        //this.GetType().GetMethod("Start", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).Invoke(this, new object[] { });
        targetsTransform[0].position += Vector3.up * 3f;

        targetMeshRenderer.material = blueMat;

        targetMeshRenderer.material.color = new Color(0f, 1f, 1f, 1f);

        

    }
	
	// Update is called once per frame
	void Update () {
	
	}
}
