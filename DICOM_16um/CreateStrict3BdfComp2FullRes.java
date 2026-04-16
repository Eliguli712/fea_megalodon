import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CreateStrict3BdfComp2FullRes {
  private static final String MPH = "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static String ts() {
    return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
  }

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) if (needle.equals(t)) return true;
    return false;
  }

  private static String safeMsg(Throwable t) {
    if (t == null) return "";
    String m = t.getMessage();
    if (m == null || m.isEmpty()) return t.getClass().getSimpleName();
    return m.replace('\n', ' ').replace('\r', ' ');
  }

  private static void configureCompMeshFromMpart2(Model m, String compTag, String meshTag, List<String> logs) {
    if (!hasTag(m.component(compTag).mesh().tags(), meshTag)) {
      logs.add("COMP_MESH_SKIP|" + compTag + "/" + meshTag + "|reason=missing");
      return;
    }
    try {
      if (!hasTag(m.component(compTag).mesh(meshTag).feature().tags(), "impmsh")) {
        m.component(compTag).mesh(meshTag).feature().create("impmsh", "Import");
      }
      m.component(compTag).mesh(meshTag).feature("impmsh").set("source", "sequence");
      m.component(compTag).mesh(meshTag).feature("impmsh").set("sequence", "mpart2");
      try { m.component(compTag).mesh(meshTag).feature("impmsh").set("buildsource", "on"); } catch (Exception ignored) {}
      try { m.component(compTag).mesh(meshTag).feature("impmsh").set("domelemsequence", "on"); } catch (Exception ignored) {}
      m.component(compTag).mesh(meshTag).run("impmsh");
      logs.add("COMP_MESH_RUN|" + compTag + "/" + meshTag + "/impmsh|ok=true");
    } catch (Exception e) {
      logs.add("COMP_MESH_RUN|" + compTag + "/" + meshTag + "/impmsh|ok=false|err=" + safeMsg(e));
    }

    try {
      int tet = m.component(compTag).mesh(meshTag).getNumElem("tet");
      logs.add("COMP_MESH_TET|" + compTag + "/" + meshTag + "|" + tet);
    } catch (Exception e) {
      logs.add("COMP_MESH_TET|" + compTag + "/" + meshTag + "|ERR|" + safeMsg(e));
    }
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);
    logs.add("COMPONENTS_BEFORE|" + Arrays.toString(m.component().tags()));

    try {
      if (hasTag(m.component().tags(), "comp2")) {
        m.component().remove("comp2");
        logs.add("COMP_REMOVE|comp2|ok=true");
      }
    } catch (Exception e) {
      logs.add("COMP_REMOVE|comp2|ok=false|err=" + safeMsg(e));
    }

    try {
      m.component().duplicate("comp2", "comp1");
      logs.add("COMP_DUPLICATE|comp1->comp2|ok=true");
    } catch (Exception e) {
      logs.add("COMP_DUPLICATE|comp1->comp2|ok=false|err=" + safeMsg(e));
      String backup = MPH + ".bak-" + ts();
      m.save(backup);
      m.save(MPH);
      for (String line : logs) System.out.println(line);
      return;
    }

    try {
      m.component("comp2").label("comp2_full_resolution_mesh2");
    } catch (Exception ignored) {}

    configureCompMeshFromMpart2(m, "comp2", "mesh1", logs);
    configureCompMeshFromMpart2(m, "comp2", "mesh2", logs);

    logs.add("COMPONENTS_AFTER|" + Arrays.toString(m.component().tags()));

    String backup = MPH + ".bak-" + ts();
    m.save(backup);
    logs.add("BACKUP|" + backup);
    m.save(MPH);
    logs.add("SAVED|" + MPH);

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
