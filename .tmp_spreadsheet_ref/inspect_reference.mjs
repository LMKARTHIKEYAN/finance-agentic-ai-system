import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "D:/Agentic AI Project/GP_percent_variance_bridge.xlsx";
const input = await FileBlob.load(source);
const workbook = await SpreadsheetFile.importXlsx(input);
const overview = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 5000,
});
const logs = ["SHEETS", overview.ndjson];

for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange();
  if (!used) continue;
  const region = await workbook.inspect({
    kind: "region",
    sheetId: sheet.name,
    range: used.address,
    maxChars: 12000,
    tableMaxRows: 40,
    tableMaxCols: 20,
  });
  logs.push(`REGION ${sheet.name} ${used.address}`, region.ndjson);
  const formulas = await workbook.inspect({
    kind: "formula",
    sheetId: sheet.name,
    range: used.address,
    maxChars: 8000,
    options: { maxResults: 120 },
  });
  logs.push(`FORMULAS ${sheet.name}`, formulas.ndjson);
  const preview = await workbook.render({
    sheetName: sheet.name,
    range: used.address,
    scale: 1.5,
    format: "png",
  });
  await fs.writeFile(
    `.tmp_spreadsheet_ref/gp_${sheet.name.replace(/[^a-z0-9]/gi, "_")}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
await fs.writeFile(".tmp_spreadsheet_ref/gp_inspection.txt", logs.join("\n"), "utf8");
