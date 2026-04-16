import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;

public class ProbeCurrentModelState {
  private static void p(String s) { System.out.println(s); }

  private static void printSelectionInfo(Model m, String selTag) {
    try {
      Selection sel = m.component("comp1").selection(selTag);
      try {
        int[] ent = sel.entities(3);
        p("selection " + selTag + " dim3 count=" + (ent == null ? -1 : ent.length));
      } catch (Exception e) { p("selection " + selTag + " dim3 read failed: " + e.getMessage()); }
      try {
        int[] ent = sel.entities(2);
        p("selection " + selTag + " dim2 count=" + (ent == null ? -1 : ent.length));
      } catch (Exception e) { p("selection " + selTag + " dim2 read failed: " + e.getMessage()); }
    } catch (Exception e) {
      p("selection " + selTag + " missing");
    }
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      p("studies=" + Arrays.toString(m.study().tags()));
    } catch (Exception e) { p("study tags failed: " + e.getMessage()); }

    try {
      p("solvers=" + Arrays.toString(m.sol().tags()));
    } catch (Exception e) { p("solver tags failed: " + e.getMessage()); }

    try {
      p("datasets=" + Arrays.toString(m.result().dataset().tags()));
    } catch (Exception e) { p("dataset tags failed: " + e.getMessage()); }

    try {
      p("solid features=" + Arrays.toString(m.component("comp1").physics("solid").feature().tags()));
    } catch (Exception e) { p("solid feature tags failed: " + e.getMessage()); }

    try {
      p("mesh tags comp1=" + Arrays.toString(m.component("comp1").mesh().tags()));
      for (String mt : m.component("comp1").mesh().tags()) {
        try { p("mesh " + mt + " features=" + Arrays.toString(m.component("comp1").mesh(mt).feature().tags())); }
        catch (Exception ex) { p("mesh " + mt + " feature read failed: " + ex.getMessage()); }
      }
    } catch (Exception e) { p("mesh read failed: " + e.getMessage()); }

    try {
      p("geom part tags=" + Arrays.toString(m.geom().tags()));
      if (Arrays.asList(m.geom().tags()).contains("part1")) {
        p("part1 features=" + Arrays.toString(m.geom("part1").feature().tags()));
      }
    } catch (Exception e) { p("geom tags failed: " + e.getMessage()); }

    printSelectionInfo(m, "sel_snout");
    printSelectionInfo(m, "sel_snout2");
    printSelectionInfo(m, "sel_tail_fix");

    try {
      p("bndl1 forceType=" + m.component("comp1").physics("solid").feature("bndl1").getString("forceType"));
      p("bndl1 force=" + Arrays.toString(m.component("comp1").physics("solid").feature("bndl1").getStringArray("force")));
    } catch (Exception e) { p("bndl1 read failed: " + e.getMessage()); }

    try {
      p("solid selection dim3 count=" + m.component("comp1").physics("solid").selection().entities(3).length);
    } catch (Exception e) { p("solid selection read failed: " + e.getMessage()); }

    for (String st : new String[]{"std1","std_nh","std_og","std_mr2","std_mr5","std_pr"}) {
      try {
        m.study(st);
        p("study " + st + " exists");
        try {
          String[][] meshMap = m.study(st).feature("stat").getStringMatrix("mesh");
          if (meshMap != null) p("study " + st + " mesh map rows=" + meshMap.length);
        } catch (Exception e) { p("study " + st + " mesh map read failed: " + e.getMessage()); }
      } catch (Exception e) {
        p("study " + st + " missing");
      }
    }
  }
}
