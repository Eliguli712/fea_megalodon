import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.Arrays;

public class ProbeStudyStatProps {
  public static void main(String[] args) throws Exception {
    Model m = ModelUtil.load("Model", "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph");
    String[] studies = m.study().tags();
    for (String st : studies) {
      if (!Arrays.asList(m.study(st).feature().tags()).contains("stat")) continue;
      PropFeature pf = m.study(st).feature("stat");
      System.out.println("STUDY|" + st + "|props=" + Arrays.toString(pf.properties()));
      for (String k : new String[]{"mesh","geomselection","physselection","activate","geometricNonlinearity","plot"}) {
        try {
          String t = pf.getValueType(k);
          String s = "";
          try { s = pf.getString(k); } catch (Exception ignored) {}
          String[] arr = null;
          try { arr = pf.getStringArray(k); } catch (Exception ignored) {}
          String[][] mat = null;
          try { mat = pf.getStringMatrix(k); } catch (Exception ignored) {}
          System.out.println("  KEY|" + k + "|type=" + t + "|s=" + s + "|arr=" + Arrays.toString(arr) + "|mat=" + (mat==null?"null":Arrays.deepToString(mat)));
        } catch (Exception e) {
          System.out.println("  KEY|" + k + "|ERR=" + e.getMessage());
        }
      }
    }
  }
}
