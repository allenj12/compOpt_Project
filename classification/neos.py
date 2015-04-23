import xmlrpclib
import sys

NEOS_HOST = "neos-server.org"
NEOS_PORT = 3332

neos = xmlrpclib.Server("http://%s:%d" % (NEOS_HOST, NEOS_PORT))
solver_template = neos.getSolverTemplate("nco", "MINOS", "AMPL")

xml_input = ""
xml_input += "<document>\n"
xml_input += "<category>nco</category>\n"
xml_input += "<solver>MINOS</solver>\n"
xml_input += "<inputMethod>AMPL</inputMethod>\n\n"
xml_input += "<model><![CDATA[\n"

model_file = open("HW3_QSARSVM_mod.txt")
xml_input += model_file.read()
xml_input += "]]></model>\n"

xml_input += "<data><![CDATA[\n"
data_file = open("HW3_QSARSVM_dat.txt")
xml_input += data_file.read()
xml_input += "]]></data>\n"


xml_input += "<commands><![CDATA[\n"
command_file = open("HW3_QSARSVM_cmd.txt")
xml_input += command_file.read()
xml_input += "]]></commands>\n"

xml_input += "</document>"
f = open("output", "w")
f.write(xml_input)


(jobNumber, password) = neos.submitJob(xml_input)
if jobNumber == 0:
    print password
offset = 0

status = ""
#Print out partial job output while job is running
while status != "Done":
    (msg, offset) = neos.getIntermediateResults(jobNumber, password, offset)

    status = neos.getJobStatus(jobNumber, password)

#Print out the final result
msg = neos.getFinalResults(jobNumber, password).data
print msg
#sys.stdout.write(msg)
