resource "azurerm_storage_container" "scania_storager_container" {
  name                  = var.scania_storage_container_name
  storage_account_name  = azurerm_storage_account.scania_storage_account.name
  container_access_type = var.scania_container_access_type
}