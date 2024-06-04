using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.XR.Interaction.Toolkit;
using System.Collections.Generic;
using System.IO;

public class get6Dof : MonoBehaviour
{
    // public Text positionText;
    GameObject text;
    public GameObject headset;
    public GameObject leftController;
    public GameObject rightController;
    private string fileName; // file save place
    private List<string> data_string_all;  // data buffer
    float delta_time;
    void Start()
    {
        delta_time = 0.0f;
        text = GameObject.Find("canvasText");
        data_string_all = new List<string>();
        string randomFloat = Random.value.ToString("F5").Substring(2);  // "F5" means 5 decimal places
        fileName = Application.persistentDataPath + "/" + randomFloat + ".txt";
        // Debug.Log(fileName);
        while(File.Exists(fileName)){
            randomFloat = Random.value.ToString("F5").Substring(2);
            fileName = Application.persistentDataPath + "/" + randomFloat + ".txt";
        }
    }

    // Update is called once per frame
    void Update()
    {
        // calculate FPS
        delta_time += (Time.unscaledDeltaTime - delta_time) * 0.1f;
        float fps = 1.0f / delta_time;
        // Debug.Log("FPS:" + fps);

        // These 2 expressions are equivalent.
        // Vector3 headsetPosition = Camera.main.transform.position;
        // Quaternion headsetRotation = Camera.main.transform.rotation;
        Vector3 headsetPosition2 = headset.transform.position;
        Quaternion headsetRotation2 = headset.transform.rotation;

        Vector3 leftControllerPosition = leftController.transform.position;
        Quaternion leftControllerRotation = leftController.transform.rotation;
        // Quaternion leftControllerRotation = leftController.transform.rotation * Quaternion.Euler(new Vector3(-180f, -90f, -80f));

        Vector3 rightControllerPosition = rightController.transform.position;
        Quaternion rightControllerRotation = rightController.transform.rotation;
        // Quaternion rightControllerRotation = rightController.transform.rotation * Quaternion.Euler(new Vector3(0f, 90f, -80f));

        // data to show
        // string data = "HeadsetPosition:"+ xrOriginPosition.ToString("F4") + ";"  +  "\n" +
        //               "HeadsetRotation: " + xrOriginRotation.eulerAngles.ToString("F4") + ";" ;
        // "F4" means 4 decimal places
        // Save the Quaternion
        string vis_data = "FPS:" + fps + "\n" +
                      "Headset-Position: " + headsetPosition2.ToString("F4") + "\n" +
                      "Headset-Rotation: " + headsetRotation2.ToString("F4") + "\n" +
                      "LeftController-Position: " + leftControllerPosition.ToString("F4") + "\n" +
                      "LeftController-Rotation: " + leftControllerRotation.ToString("F4") + "\n" +
                      "RightController-Position: " + rightControllerPosition.ToString("F4") + "\n" +
                      "RightController-Rotation: " + rightControllerRotation.ToString("F4");

        // Debug.Log("Quaternion values: x=" + leftControllerRotation.x + ", y=" + leftControllerRotation.y +
        // ", z=" + leftControllerRotation.z + ", w=" + leftControllerRotation.w);

        // positionText.text = data;
        text.GetComponent<TextMeshProUGUI>().text = vis_data;

        // data to save, separated by ;
        string save_data =
                      "FPS:" + fps + ";" +
                      "Headset-Position: " + headsetPosition2.ToString("F4") + ";" +
                      "Headset-Rotation: " + headsetRotation2.ToString("F4") + ";" +
                      "LeftController-Position: " + leftControllerPosition.ToString("F4") + ";" +
                      "LeftController-Rotation: " + leftControllerRotation.ToString("F4") + ";" +
                      "RightController-Position: " + rightControllerPosition.ToString("F4") + ";" +
                      "RightController-Rotation: " + rightControllerRotation.ToString("F4")+ ";" + "\n";

        if(data_string_all.Count < 200){
            data_string_all.Add(save_data);
        }else{
            string temp_data = string.Join("\n", data_string_all);
            File.AppendAllText(fileName, temp_data);
            data_string_all.Clear();
        }


    }

}