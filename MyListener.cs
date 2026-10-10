using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UnityRobotReceiver : MonoBehaviour
{
    [System.Serializable]
    public class JointTarget
    {
        public string jointName;
        [HideInInspector] public ArticulationBody body;
    }

    [Header("Configurações do Robô")]
    public JointTarget[] joints = new JointTarget[]
    {
        new JointTarget { jointName = "Link_1_1" },
        new JointTarget { jointName = "Link_2_1" },
        new JointTarget { jointName = "Link_3_1" },
        new JointTarget { jointName = "Link_4_1" },
        new JointTarget { jointName = "Link_5_1" },
        new JointTarget { jointName = "Link_6_1" },
    };

    [Header("Configurações de Rede")]
    public int connectionPort = 25001;

    [Header("Controle de Movimento")]
    public float maxDegreesPerSecond = 60f;
    public bool manualMode = false; 

    private Thread thread;
    private TcpListener server;
    private TcpClient client;
    private bool running;

    private readonly Queue<float[]> waypointQueue = new Queue<float[]>();
    private readonly object lockObject = new object();
    private bool hasActiveSegment = false;

    private float[] segmentStartAngles = new float[6];
    private float[] segmentEndAngles = new float[6];
    private float segmentDuration = 0f;
    private float segmentElapsed = 0f;

    private float[] commandedAngles = new float[6];
    private bool initializedUnwrap = false;

    private float[] currentJointAngles = new float[6];
    private bool triggerSendButton = false;

    private string[] angleInputs = new string[6] { "0", "0", "0", "0", "0", "0" };

    private float[] minLimits = new float[6] { -200f, -75f, -50f, -200f, -200f, -200f };
    private float[] maxLimits = new float[6] { 200f, 125f, 100f, 200f, 200f, 200f };

    void Start()
    {
        Application.runInBackground = true; //[cite: 1]

        foreach (var j in joints)
        {
            Transform t = FindDeepChild(transform, j.jointName); //[cite: 1]
            if (t != null)
            {
                j.body = t.GetComponent<ArticulationBody>(); //[cite: 1]
                if (j.body == null)
                    Debug.LogWarning($"[Receiver] '{j.jointName}' encontrado, mas sem ArticulationBody."); //[cite: 1]
            }
        }

        running = true;
        thread = new Thread(new ThreadStart(StartServer)); //[cite: 1]
        thread.Start(); //[cite: 1]
    }

    void StartServer()
    {
        try
        {
            server = new TcpListener(IPAddress.Any, connectionPort);
            server.Start();
            Debug.Log($"[Servidor Unity] Aguardando conexão na porta {connectionPort}...");

            client = server.AcceptTcpClient();
            Debug.Log("[Servidor Unity] Cliente Python conectado!");

            NetworkStream stream = client.GetStream();
            System.IO.StreamReader reader = new System.IO.StreamReader(stream, Encoding.UTF8);
            System.IO.StreamWriter writer = new System.IO.StreamWriter(stream, Encoding.UTF8) { AutoFlush = true };

            while (running)
            {
                if (client.Connected)
                {
                    // 1. Lê os comandos que vêm do Python
                    if (stream.DataAvailable)
                    {
                        string dataReceived = reader.ReadLine();
                        if (!string.IsNullOrEmpty(dataReceived))
                        {
                            ParseAndStoreAngles(dataReceived);
                        }
                    }

                    // 2. Envia os 6 ângulos das juntas + 1 flag do botão para o Python (Formato: "J1,J2,J3,J4,J5,J6,Gatilho")
                    string anglesMessage = "";
                    lock (lockObject)
                    {
                        // DENTRO DO StartServer(), NO BLOCO lock (lockObject):

                        int flag = triggerSendButton ? 1 : 0;

                        // Força o C# a formatar os floats usando ponto (.) em vez de vírgula (,)
                        string[] formattedAngles = Array.ConvertAll(currentJointAngles, a => a.ToString(CultureInfo.InvariantCulture));
                        anglesMessage = string.Join(",", formattedAngles) + "," + flag;

                        triggerSendButton = false; // Reseta a flag após o envio
                    }
                    writer.WriteLine(anglesMessage);
                }
                Thread.Sleep(20); // ~50Hz
            }
        }
        catch (Exception e) 
        { 
            Debug.LogError($"Erro na rede (Unity Server): {e.Message}"); 
        }
    }

    void ParseAndStoreAngles(string data)
    {
        Debug.Log($"[Receiver] Angulos recebidos: {data}");
        try
        {
            string[] splitData = data.Trim().Split(',');
            if (splitData.Length >= 6)
            {
                float[] waypoint = new float[splitData.Length]; 
                for (int i = 0; i < splitData.Length; i++)
                {
                    waypoint[i] = float.Parse(splitData[i], CultureInfo.InvariantCulture); //[cite: 1]
                }

                lock (lockObject)
                {
                    waypointQueue.Enqueue(waypoint);
                }
            }
        }
        catch (Exception ex) { Debug.LogWarning($"Erro no Parse: {ex.Message}"); } //[cite: 1]
    }

    void OnGUI()
    {
        int boxWidth = 320;
        int boxHeight = 70 + (joints.Length * 60) + 50;

        GUI.Box(new Rect(10, 10, boxWidth, boxHeight), "Painel de Controle do Robô");

        manualMode = GUI.Toggle(new Rect(20, 35, 250, 20), manualMode, " Controle Manual (Ignorar Python)");

        int yPos = 70;
        for (int i = 0; i < joints.Length; i++)
        {
            yPos = 70 + (i * 60);

            // Limites configurados para a junta atual
            float minVal = minLimits[i];
            float maxVal = maxLimits[i];

            if (manualMode)
            {
                // Rótulo da Junta
                GUI.Label(new Rect(20, yPos, 200, 20), $"{joints[i].jointName}:");

                // 1. Slider com limites individuais
                float sliderVal = GUI.HorizontalSlider(new Rect(20, yPos + 22, 200, 20), commandedAngles[i], minVal, maxVal);

                // Se moveu o slider, atualiza a caixa de texto
                if (Mathf.Abs(sliderVal - commandedAngles[i]) > 0.001f)
                {
                    commandedAngles[i] = sliderVal;
                    angleInputs[i] = sliderVal.ToString("F1", CultureInfo.InvariantCulture);
                }

                // 2. Caixa de Texto para Input Numérico
                if (string.IsNullOrEmpty(angleInputs[i]))
                {
                    angleInputs[i] = commandedAngles[i].ToString("F1", CultureInfo.InvariantCulture);
                }

                string textVal = GUI.TextField(new Rect(230, yPos + 18, 60, 22), angleInputs[i]);

                // Se digitou na caixa, limita dentro da faixa definida para esta junta
                if (textVal != angleInputs[i])
                {
                    angleInputs[i] = textVal;
                    if (float.TryParse(textVal, NumberStyles.Any, CultureInfo.InvariantCulture, out float parsedVal))
                    {
                        commandedAngles[i] = Mathf.Clamp(parsedVal, minVal, maxVal);
                    }
                }
            }
            else
            {
                // Modo Leitura (Exibe posição vinda do Python/Física)
                float displayValue = currentJointAngles[i];
                GUI.Label(new Rect(20, yPos, 280, 20), $"{joints[i].jointName}: {displayValue:F1}°");
                GUI.HorizontalSlider(new Rect(20, yPos + 22, 270, 20), displayValue, minVal, maxVal);
            }
        }

        // Botão de Envio para o Python
        if (GUI.Button(new Rect(20, yPos + 55, 270, 40), "ENVIAR PARA O PYTHON"))
        {
            lock (lockObject)
            {
                triggerSendButton = true;
            }
            Debug.Log("[Unity] Comando de envio disparado para o Python!");
        }
    }

    void FixedUpdate()
    {
        lock (lockObject)
        {
            if (!initializedUnwrap)
            {
                for (int i = 0; i < joints.Length; i++)
                {
                    float startAngle = joints[i].body != null ? joints[i].body.xDrive.target : 0f; //[cite: 1]
                    segmentEndAngles[i] = startAngle; //[cite: 1]
                    commandedAngles[i] = startAngle; //[cite: 1]
                }
                initializedUnwrap = true; //[cite: 1]
            }

            // Atualiza as posições reais das juntas em graus (usadas no envio para o Python)
            for (int i = 0; i < joints.Length; i++)
            {
                if (joints[i].body != null)
                {
                    currentJointAngles[i] = (float)Math.Round(commandedAngles[i], 2);
                }
            }

            // --- MODO MANUAL: Aplica os sliders diretamente na física ---
            if (manualMode)
            {
                waypointQueue.Clear();      
                hasActiveSegment = false;   
                
                for (int i = 0; i < joints.Length; i++)
                {
                    if (joints[i].body == null) continue;
                    ArticulationDrive drive = joints[i].body.xDrive;
                    drive.target = commandedAngles[i];
                    joints[i].body.xDrive = drive;
                }
                return; 
            }

            // --- MODO AUTOMÁTICO (PYTHON): Interpolação dos waypoints recebidos ---
            if (!hasActiveSegment && waypointQueue.Count > 0)
            {
                float[] waypoint = waypointQueue.Dequeue();
                float sumSquares = 0f;

                for (int i = 0; i < joints.Length && i < 6; i++)
                {
                    segmentStartAngles[i] = commandedAngles[i]; //[cite: 1]

                    float wrappedNew = waypoint[i]; //[cite: 1]
                    float wrappedPrevious = Mathf.Repeat(segmentEndAngles[i] + 180f, 360f) - 180f; //[cite: 1]
                    float delta = Mathf.DeltaAngle(wrappedPrevious, wrappedNew); //[cite: 1]
                    segmentEndAngles[i] = segmentEndAngles[i] + delta; //[cite: 1]

                    sumSquares += delta * delta;
                }

                float totalDistance = Mathf.Sqrt(sumSquares);
                float feedrate = waypoint.Length > 6 ? waypoint[6] : -1f;

                if (feedrate > 0f)
                {
                    segmentDuration = totalDistance > 0.001f
                        ? Mathf.Max((totalDistance / feedrate) * 60f, 0.02f)
                        : 0.02f;
                }
                else
                {
                    float maxDelta = 0f;
                    for (int i = 0; i < joints.Length && i < 6; i++)
                    {
                        maxDelta = Mathf.Max(maxDelta, Mathf.Abs(segmentEndAngles[i] - segmentStartAngles[i]));
                    }
                    segmentDuration = Mathf.Max(maxDelta / maxDegreesPerSecond, 0.02f);
                }

                segmentElapsed = 0f; //[cite: 1]
                hasActiveSegment = true; //[cite: 1]
            }

            if (!hasActiveSegment) return; //[cite: 1]

            segmentElapsed += Time.fixedDeltaTime; //[cite: 1]
            float t = Mathf.Clamp01(segmentElapsed / segmentDuration); //[cite: 1]

            for (int i = 0; i < joints.Length; i++)
            {
                if (joints[i].body == null) continue; //[cite: 1]

                commandedAngles[i] = Mathf.Lerp(segmentStartAngles[i], segmentEndAngles[i], t); //[cite: 1]

                ArticulationDrive drive = joints[i].body.xDrive; //[cite: 1]
                drive.target = commandedAngles[i]; //[cite: 1]
                joints[i].body.xDrive = drive; //[cite: 1]
            }

            if (t >= 1f)
            {
                hasActiveSegment = false; //[cite: 1]
            }
        }
    }

    private Transform FindDeepChild(Transform parent, string name)
    {
        foreach (Transform child in parent) //[cite: 1]
        {
            if (child.name == name) return child; //[cite: 1]
            Transform result = FindDeepChild(child, name); //[cite: 1]
            if (result != null) return result; //[cite: 1]
        }
        return null; //[cite: 1]
    }

    void OnApplicationQuit()
    {
        running = false; //[cite: 1]
        if (client != null) client.Close(); //[cite: 1]
        if (server != null) server.Stop(); //[cite: 1]
        if (thread != null) thread.Abort(); //[cite: 1]
    }
}