import java.io.FileReader;

import static java.lang.System.out;

public class RunJson {

    public static void main(String args[]) {
        byte numStates = 30;
        int pass = numStates / 2;
        boolean numActionsPerState = true;
        double gamma = 0.75D;
        boolean count = false;

        try {
            String fname = args[0];
            FileReader s = new FileReader(fname);
            StringBuffer sb = new StringBuffer();
            int c = s.read();
            while (c != -1) {
                sb.append((char) c);
                c = s.read();
            }
            String data = sb.toString();
            out.println(data);
            JsonToBurlap jtb = new JsonToBurlap(data);
            LongPolicyIterationGraderTestRunVersion lpigtrv = new LongPolicyIterationGraderTestRunVersion();
            int iterations = lpigtrv.countPIIterations(jtb.getGraphDefinedDomain().generateDomain(), jtb.numStates, jtb.getRF(), jtb.getTF(), jtb.gamma);
            out.println(String.format("Number of iterations: %d", iterations));
        } catch (IndexOutOfBoundsException e) {
            out.println("Input file not provided");
        } catch (Exception e) {
            out.println(String.format("Failed to read file: %s", args[0]));
        }

    }
}
