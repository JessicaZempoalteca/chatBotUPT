import BotGif from '../components/BotGif';

let name = ""

export const STEPS = [
	{
		id: '1',
		component: <BotGif />,
		trigger: '2',
	},
	{
		id: '2',
		message: 'Hola mi nombre es AsAD, ¿Cuál es tu nombre?',
		trigger: '3',
	},
	{
		id: '3',
		user: true,
		trigger: '4',
	},
	{
		id: '4',
		message: ({previousValue}) => {
			name = previousValue
			return 'Hola {previousValue}. Soy tu asesor académico sobre el tema de Ciberseguridad!'
		},
		trigger: '5',
	},
	{
		id: '5',
		message:
			'Voy a responder tus dudas sobre este tema en un nivel Introductorio. Para ayudarte, dividimos la temática en 3 apartados',
		trigger: '6',
	},
	{
		id: '6',
		options: [
			{ value: 1, label: 'Ciberseguridad 🖥', trigger: '7' },
			{ value: 2, label: 'Ataques 💣', trigger: '22' },
			{ value: 3, label: 'Protegerse 😷', trigger: '45' },
			{ value: 4, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	//Topics about cibersecurity
	{
		id: '7',
		options: [
			{ value: 1, label: 'Ciberseguridad', trigger: '8' },
			{ value: 2, label: 'Hacker', trigger: '9' },
			{ value: 3, label: 'Ciberdelincuente', trigger: '10' },
			{ value: 4, label: 'Hacker sombrero blanco', trigger: '11' },
			{ value: 5, label: 'Hacker sombrero gris', trigger: '12' },
			{ value: 6, label: 'Amenaza', trigger: '13' },
			{ value: 7, label: 'Vulnerabilidad', trigger: '14' },
			{ value: 8, label: 'Red de Datos', trigger: '15' },
			{ value: 9, label: 'Hacking', trigger: '16' },
			{ value: 10, label: 'Cracker', trigger: '17' },
			{ value: 11, label: 'Exploit', trigger: '18' },
			{ value: 12, label: 'Hacktivista', trigger: '19' },
			{ value: 13, label: 'Síntomas de contagio', trigger: '20' },
			{ value: 14, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	{
		id: "8",
		message: "Es la práctica de proteger sistemas, redes y programas de ataques digitales. Garantizando la Confidencialidad, Integridad y Disponibilidad de los datos.",
		trigger: "21" 
	},
	{
		id: "9",
		message: "Persona con grandes conocimientos de informática que se dedica a detectar fallos de seguridad en sistemas informáticos.",
		trigger: "21"
	},
	{
		id: "10",
		message: "Tambien llamado Hacker sombrero negro. Son personas que intentan aprovechar las vulnerabilidades de los sistemas para obtener una ganancia personal o financiera",
		trigger: "21"
	},
	{
		id: "11",
		message: "Conocido como Hacker Ético u Ethical Hacker. Son personas ingresan a las redes o los sistemas informáticos para descubrir las debilidades para poder mejorar la seguridad de estos sistemas con permiso previo.",
		trigger: "21"
	},
	{
		id: "12",
		message: "Persona que se proponerse encontrar vulnerabilidades en un sistema, pero solo informarán de sus hallazgos a los propietarios de un sistema si hacerlo coincide con sus intereses.",
		trigger: "21"
	},
	{
		id: "13",
		message: "Es una circunstancia que tiene el potencial de causar daños o pérdidas. Las amenazas pueden dar lugar a un ataque en el equipo.",
		trigger: "21"
	},
	{
		id: "14",
		message: "Es una debilidad del sistema informático que puede ser utilizada para causar un daño.",
		trigger: "21"
	},
	{
		id: "15",
		message: "Conjunto de dispositivos finales, de red y medios físicos y lógicos que tiene la finalidad comunicar datos de manera local o remota.",
		trigger: "21"
	},
	{
		id: "16",
		message: "Conjunto de técnicas a través de las cuales se accede a un sistema informático vulnerando las medidas de seguridad establecidas originariamente.",
		trigger: "21"
	},
	{
		id: "17",
		message: "El término proviene del vocablo inglés crack (romper). Es alguien que viola la seguridad de un sistema de forma similar a un hacker, solo que ilegalmente y con diferentes fines.",
		trigger: "21"
	},
	{
		id: "18",
		message: "Programa escrito para aprovechar una vulnerabilidad de seguridad conocida.",
		trigger: "21"
	},
	{
		id: "19",
		message: "Persona o personas que protestan con fines de activismo político, social o económico en forma de ataques cibernéticos.",
		trigger: "21"
	},
	{
		id: "20",
		message: "Dispositivo lento, equipo se congela o bloquea con frecuencia, Archivos modificados o eliminados, Archivos desconocidos, Ejecución de servicios desconocidos, Envíos de correos sin autorización. ",
		trigger: "21"
	},
	{
		id: "21",
		options: [
			{ value: 1, label: 'Ver otro concepto sobre Ciberseguridad', trigger: '7' },
			{ value: 2, label: 'Ver temas', trigger: '6' },
			{ value: 3, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	//End topics of cibersecurity
	//Topics about attacks
	{
		id: '22',
		options: [
			{ value: 1, label: '¿Por qué un ataque?', trigger: '23' },
			{ value: 2, label: '¿Qué quieren los ciberdelincuentes?', trigger: '24' },
			{ value: 3, label: 'Malware', trigger: '25' },
			{ value: 4, label: 'Spyware', trigger: '26' },
			{ value: 5, label: 'Adware', trigger: '27' },
			{ value: 6, label: 'Ransomware', trigger: '28' },
			{ value: 7, label: 'Scareware', trigger: '29' },
			{ value: 8, label: 'Rootkit', trigger: '30' },
			{ value: 9, label: 'Virus', trigger: '31' },
			{ value: 10, label: 'Troyano', trigger: '32' },
			{ value: 11, label: 'Gusano', trigger: '33' },
			{ value: 12, label: 'DoS', trigger: '34' },
			{ value: 13, label: 'DDoS', trigger: '35' },
			{ value: 14, label: 'Spoofing', trigger: '36' },
			{ value: 15, label: 'Sniffing', trigger: '37' },
			{ value: 16, label: 'Snooping', trigger: '38' },
			{ value: 17, label: 'Man in the middle', trigger: '39' },
			{ value: 18, label: 'Flooding', trigger: '40' },
			{ value: 19, label: 'Keylogger', trigger: '41' },
			{ value: 20, label: 'Phishing', trigger: '42' },
			{ value: 21, label: 'Suplantación de Identidad', trigger: '43' },
			{ value: 22, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	{
		id: "23",
		message: "Para obtener beneficios. Principalmente económicos.",
		trigger: "44"
	},
	{
		id: "24",
		message: "Información personal, Dinero. Reconocimiento.",
		trigger: "44"
	},
	{
		id: "25",
		message: "Tambien llamado Software Malicioso. Es cualquier tipo de software diseñado para obtener acceso no autorizado, robar datos, causar daños o comprometer un sistema. Virus, Gusanos, troyanos, etc. ",
		trigger: "44"
	},
	{
		id: "26",
		message: "Software espía, monitorea su actividad en línea y puede registrar cada tecla que presiona en el teclado, así como capturar casi todos sus datos, incluida información personal confidencial, como sus datos bancarios en línea",
		trigger: "44"
	},
	{
		id: "27",
		message: "Con frecuencia se instala con algunas versiones de software y está diseñado para entregar automáticamente anuncios a un usuario, la mayoría de las veces en un navegador web. Aparecen anuncios emergentes constantes en la pantalla.",
		trigger: "44"
	},
	{
		id: "28",
		message: "Es un tipo de software malicioso. Está diseñado para exigir dinero mediante el bloqueo del acceso a los archivos o el sistema informático hasta que se pague un rescate. El pago del rescate no garantiza que se recuperen los archivos o se restaure el sistema.",
		trigger: "44"
	},
	{
		id: "29",
		message: "Consiste principalmente en ventanas emergentes que aparecen para advertirle que su sistema está en riesgo y necesita ejecutar un programa específico para que vuelva a funcionar normalmente.",
		trigger: "44"
	},
	{
		id: "30",
		message: "Malware diseñado para modificar el sistema operativo para crear una puerta trasera, que los atacantes pueden usar para acceder a su computadora de forma remota. Es difícil de detectar. En la mayoría de los casos, un equipo infectado por un rootkit debe borrarse y reinstalarse el software.",
		trigger: "44"
	},
	{
		id: "31",
		message: "Es un tipo de programa informático que, cuando se ejecuta, se replica y se adjunta a otros archivos ejecutables, como un documento, insertando su propio código. Se propagan por unidades USB, discos ópticos, recursos de red compartidos o correo electrónico.",
		trigger: "44"
	},
	{
		id: "32",
		message: "Tipo de malware que lleva a cabo operaciones maliciosas al enmascarar su verdadera intención. Los troyanos aprovechan sus privilegios de usuario y se encuentran con mayor frecuencia en archivos de imagen, archivos de audio o juegos. Como el caballo de Troya.",
		trigger: "44"
	},
	{
		id: "33",
		message: "Tipo de malware que se replica a sí mismo para propagarse de un ordenador a otro. No requiere un programa anfitrión para ejecutarse, los gusanos pueden ejecutarse por sí mismos. Pueden propagarse muy rápidamente por la red. Son responsables de algunos de los ataques más devastadores en Internet. En 2001, el gusano Code Red había infectado más de 300.000 servidores en solo 19 horas.",
		trigger: "44"
	},
	{
		id: "34",
		message: "Conocido como Denegación de Servicio o Denial-of-service. Es un ataque a un sistema que causa que un servicio o recurso sea inaccesible a los usuarios legítimos. Sobre carga desde una IP.",
		trigger: "44"
	},
	{
		id: "35",
		message: "Llamado Denegación de Servicio Distribuido o Distributed Denial-of-service. Se provocan al generar grandes cantidades de información desde varios puntos de forma voluntaria. Sobre carga del servidor desde varias IPs.",
		trigger: "44"
	},
	{
		id: "36",
		message: "El atacante hace uso de técnicas para hacerse pasar por una entidad distinta a través de la falsificación de datos en una comunicación. Suplanta la identidad de alguien más.",
		trigger: "44"
	},
	{
		id: "37",
		message: "Ataque utilizado para escuchar todo lo que ocurre dentro de una red. El atacante se conoce como sniffer. El atacante se dedica a rastrear la red.",
		trigger: "44"
	},
	{
		id: "38",
		message: "Es acceso no autorizado a los datos de otra persona o compañía. Es similar al eavesdropping (ataque de espionaje). El atacante usa software para monitorear de manera remota la actividad de alguien en una red.",
		trigger: "44"
	},
	{
		id: "39",
		message: "El ataque Hombre en el Medio o MiTM. Ocurre cuando un ciberdelincuente toma el control de un dispositivo sin que el usuario lo sepa. El atacante puede interceptar y capturar información sobre el usuario antes de retransmitirla a su destino. El ataque se utiliza a menudo para robar información financiera.",
		trigger: "44"
	},
	{
		id: "40",
		message: "Tipo de ataque que inunda de peticiones al servidor",
		trigger: "44"
	},
	{
		id: "41",
		message: "Tipo de malware que registra todas las pulsaciones de teclas que realice en el ordenador. Sirve para robar información del usuario.",
		trigger: "44"
	},
	{
		id: "42",
		message: "Conjunto de técnicas para engañar a una víctima ganándose su confianza haciéndose pasar por una persona, empresa o servicio de confianza, para manipularla y hacer que realice acciones que no debería realizar.",
		trigger: "44"
	},
	{
		id: "43",
		message: "Es la práctica de enviar correos electrónicos fraudulentos que se asemejan a correos electrónicos de fuentes de buena reputación. El objetivo es robar datos sensibles, como números de tarjetas de crédito e información de inicio de sesión.",
		trigger: "44"
	},
	{
		id: "44",
		options: [
			{ value: 1, label: 'Ver otros Ataques', trigger: '22' },
			{ value: 2, label: 'Ver temas', trigger: '6' },
			{ value: 3, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	//End topic about attacks
	//Topics about protection
	{
		id: '45',
		options: [
			{ value: 1, label: 'Datos personales', trigger: '46' },
			{ value: 2, label: '¿Cómo protegerse datos personales?', trigger: '47' },
			{ value: 3, label: 'Antimalware', trigger: '48' },
			{ value: 4, label: 'Contraseñas seguras', trigger: '49' },
			{ value: 5, label: 'Ingeniería social', trigger: '50' },
			{ value: 6, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	{
		id: "46",
		message: "Nombre, fecha nacimiento, numero de seguridad social, dirección, etc.",
		trigger: "51"
	},
	{
		id: "47",
		message: "Use un antimalware, use contraseñas seguras con más de 8 caracteres (números, letras mayúsculas y minúsculas, símbolos especiales). Active en firewall. Aplique actualisasiones del sistema operativo.",
		trigger: "51"
	},
	{
		id: "48",
		message: "Software que lo protege contra virus, spyware, troyanos, adware, etc.",
		trigger: "51"
	},
	{
		id: "49",
		message: "Contraseñas con más de 8 caracteres (números, letras mayúsculas y minúsculas, símbolos especiales)",
		trigger: "51"
	},
	{
		id: "50",
		message: "Es la práctica de obtener información confidencial a través de la manipulación de usuarios legítimos muchas veces ganando su confianza ",
		trigger: "51"
	},
	{
		id: "51",
		options: [
			{ value: 1, label: 'Ver otras maneras de protegerme', trigger: '45' },
			{ value: 2, label: 'Ver temas', trigger: '6' },
			{ value: 3, label: "Salir 🚶🏻‍♂️", trigger: "52"}
		],
	},
	//End topics about protection
	{
		id: "52",
		message: () => {
			return `Espero haberte ayudado, puedes regresar cuando quieras ${name}, fue un gusto!`
		},
		end: true
	}
];
